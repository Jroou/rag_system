import asyncio
import threading
from collections.abc import AsyncIterator

from PySide6.QtCore import QObject, Signal


class AsyncBridge(QObject):
    """Persistent asyncio event loop in a background thread.
    Emits Qt signals to push results back to the UI thread."""

    token_received = Signal(str)
    stream_finished = Signal()
    stream_error = Signal(str)
    task_done = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._current_future: asyncio.Future | None = None

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, coro) -> asyncio.Future:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def run_blocking(self, func, *args, callback=None):
        """Run a blocking function in the loop's executor, emit task_done when complete."""
        async def _wrapper():
            result = await self._loop.run_in_executor(None, func, *args)
            if callback:
                self.task_done.emit(result)
            return result
        return self.submit(_wrapper())

    def start_stream(self, stream_coro):
        self._current_future = self.submit(self._consume_stream(stream_coro))

    async def _consume_stream(self, stream_coro):
        try:
            result = await stream_coro
            if isinstance(result, tuple):
                stream, sources, strategy = result
            else:
                stream = result
                sources, strategy = [], ""

            async for token in stream:
                self.token_received.emit(token)
            self.stream_finished.emit()
        except asyncio.CancelledError:
            self.stream_finished.emit()
        except Exception as e:
            self.stream_error.emit(str(e))

    def cancel_stream(self):
        if self._current_future and not self._current_future.done():
            self._current_future.cancel()

    def shutdown(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)
