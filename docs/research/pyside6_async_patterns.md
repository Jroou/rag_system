# PySide6 Async Patterns for LLM Streaming

**Status:** Decision made  
**Date:** 2026-07-15  
**Ticket:** #44

## Problem

The RAG engine exposes `aquery_stream()` returning an `AsyncIterator[str]` — each yielded string is a token from the LLM. The PySide6 desktop app must consume this iterator and append tokens to a `QTextBrowser` without blocking the Qt event loop. The ingestion pipeline (`IngestionPipeline.ingest()`) is synchronous but CPU/IO-heavy (embeddings, Qdrant upserts) and must also run without freezing the UI.

## Approaches Evaluated

### 1. qasync (asyncio event loop inside Qt)

**How it works:** Replaces the default asyncio event loop with one that cooperatively schedules coroutines inside the Qt event loop (`QEventLoop` from `qasync`). You can `await` async generators directly from slots.

| Pros | Cons |
|------|------|
| Natural `async for token in stream: widget.append(token)` | Library is community-maintained, low bus-factor |
| Single-threaded — no cross-thread signaling needed | Fragile: mixes two event loop implementations; version skew between Qt and asyncio causes subtle bugs |
| Minimal boilerplate | Debugging is harder — stack traces span both loops |
| | PySide6 6.7+ changed internal event processing; qasync patches lag behind |

**LangChain compatibility:** Direct — `aquery_stream()` works as-is inside an `async def` slot.

### 2. QThread with signals

**How it works:** Spawn a `QThread` (or use `QThreadPool` + `QRunnable`). Inside the thread, run `asyncio.run()` over the async generator. Emit a Qt signal for each token; connect the signal to a slot that updates the widget.

| Pros | Cons |
|------|------|
| Battle-tested Qt pattern; works with every PySide6 version | Requires a bridge: run a temporary asyncio loop per request inside the thread |
| Clear thread boundary — UI thread never touches async code | Signal/slot overhead per token (negligible in practice, ~μs) |
| Easy to cancel via a flag checked between tokens | More boilerplate than approach 3 |
| | Each query spins up a new event loop or keeps one alive — lifecycle management |

**LangChain compatibility:** Works, but you must create and manage an asyncio event loop inside the thread (`asyncio.new_event_loop()` + `loop.run_until_complete()`), or use the sync fallback if available.

### 3. Dedicated asyncio thread with queue (recommended)

**How it works:** One long-lived background thread runs `asyncio.run_forever()`. All async work (streaming, ingestion if made async later) is submitted via `asyncio.run_coroutine_threadsafe()`. A Qt signal bridges results back to the UI thread.

| Pros | Cons |
|------|------|
| Single asyncio loop — matches how LangChain expects to operate | Slightly more setup at app start (one thread + one loop) |
| No third-party event-loop hacks (no qasync) | Must discipline all async calls through `run_coroutine_threadsafe` |
| Clean cancellation via `Future.cancel()` | |
| Works with every PySide6 version (no Qt internals patched) | |
| Scales to multiple concurrent async tasks (ingestion + query) | |
| Signal-per-token gives smooth UI updates | |

**LangChain compatibility:** Excellent — the asyncio loop is a standard `asyncio.EventLoop`; all LangChain async primitives (`astream`, `ainvoke`, async callbacks) work without modification.

## Decision

**Use approach 3: dedicated asyncio thread with a signal bridge.**

### Rationale

1. **Stability** — no dependency on qasync's maintenance or compatibility with future PySide6 releases.
2. **LangChain native** — `aquery_stream()` returns `AsyncIterator[str]`; consuming it inside a real asyncio loop requires zero adaptation.
3. **Single loop for all async work** — both streaming queries and future async ingestion share one loop, avoiding per-request loop creation overhead.
4. **Clean cancellation** — `Future.cancel()` propagates `CancelledError` into the async generator, letting LangChain clean up HTTP connections.
5. **Testability** — the async layer can be unit-tested with `pytest-asyncio` independently of Qt.

## Code Sketch

```python
"""Minimal pattern: dedicated asyncio thread + Qt signal bridge for token streaming."""

import asyncio
import threading
from collections.abc import AsyncIterator

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QTextBrowser


class AsyncBridge(QObject):
    """Runs a persistent asyncio event loop in a background thread.
    Emits Qt signals to push results back to the UI thread."""

    token_received = Signal(str)
    stream_finished = Signal()
    stream_error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, coro) -> asyncio.Future:
        """Schedule a coroutine on the async loop from any thread."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def start_stream(self, stream_coro):
        """Submit a streaming coroutine; tokens arrive via token_received signal."""
        self._current_future = self.submit(self._consume_stream(stream_coro))

    async def _consume_stream(self, stream_coro):
        try:
            stream: AsyncIterator[str] = await stream_coro
            async for token in stream:
                # Emit signal — Qt delivers to UI thread via event queue
                self.token_received.emit(token)
            self.stream_finished.emit()
        except asyncio.CancelledError:
            self.stream_finished.emit()
        except Exception as e:
            self.stream_error.emit(str(e))

    def cancel(self):
        if hasattr(self, "_current_future"):
            self._current_future.cancel()

    def shutdown(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)


# --- Usage in the main window ---

class ChatWidget(QTextBrowser):
    def __init__(self, rag_engine, bridge: AsyncBridge):
        super().__init__()
        self._engine = rag_engine
        self._bridge = bridge
        self._bridge.token_received.connect(self._on_token)
        self._bridge.stream_finished.connect(self._on_done)
        self._bridge.stream_error.connect(self._on_error)

    @Slot(str)
    def _on_token(self, token: str):
        # Runs in UI thread — safe to update widget
        cursor = self.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(token)
        self.setTextCursor(cursor)

    @Slot()
    def _on_done(self):
        self.append("\n")  # finalize message

    @Slot(str)
    def _on_error(self, error: str):
        self.append(f"\n[Error: {error}]")

    def ask(self, query: str):
        async def _do_query():
            stream, results, strategy = await self._engine.aquery_stream(query)
            return stream

        # Wrap so _consume_stream gets the iterator
        async def _wrapped():
            return await _do_query()

        self._bridge.start_stream(_wrapped())
```

### Key implementation notes

- **One `AsyncBridge` per app lifetime** — created at startup, shut down on `QApplication.aboutToQuit`.
- **Ingestion** uses the same bridge: `bridge.submit(some_async_ingestion_coro())` — returns a `Future` you can track for progress.
- **Cancellation** — when the user clicks "Stop generating", call `bridge.cancel()`. The `CancelledError` propagates cleanly.
- **Thread safety** — never touch Qt widgets from the async thread. Signals handle the cross-thread dispatch automatically.
- **Back-pressure** — if the LLM produces tokens faster than Qt can paint (unlikely), signals queue in the Qt event loop; no data is lost.
