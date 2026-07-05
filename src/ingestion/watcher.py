import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer


class DebouncedHandler(FileSystemEventHandler):
    def __init__(
        self,
        task_queue: queue.Queue,
        supported_extensions: set[str],
        debounce_seconds: float = 2.0,
    ):
        self._queue = task_queue
        self._supported = supported_extensions
        self._debounce = debounce_seconds
        self._pending: dict[str, float] = {}
        self._lock = threading.Lock()
        self._timer_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._timer_thread.start()

    def _is_supported(self, path: str) -> bool:
        return Path(path).suffix.lower() in self._supported

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_supported(event.src_path):
            self._schedule(event.src_path, "ingest")

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_supported(event.src_path):
            self._schedule(event.src_path, "ingest")

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_supported(event.src_path):
            with self._lock:
                self._pending.pop(event.src_path, None)
            self._queue.put(("delete", event.src_path))

    def _schedule(self, path: str, action: str) -> None:
        with self._lock:
            self._pending[path] = time.time()

    def _flush_loop(self) -> None:
        while True:
            time.sleep(0.5)
            now = time.time()
            to_flush = []
            with self._lock:
                for path, timestamp in list(self._pending.items()):
                    if now - timestamp >= self._debounce:
                        to_flush.append(path)
                        del self._pending[path]
            for path in to_flush:
                self._queue.put(("ingest", path))


class FileWatcher:
    def __init__(
        self,
        folder: str,
        supported_extensions: list[str],
        on_ingest: Callable[[Path], Any],
        on_delete: Callable[[Path], Any],
        on_progress: Callable[[str], None] | None = None,
        debounce_seconds: float = 2.0,
    ):
        self._folder = Path(folder).resolve()
        self._folder.mkdir(parents=True, exist_ok=True)
        self._on_ingest = on_ingest
        self._on_delete = on_delete
        self._on_progress = on_progress
        self._queue: queue.Queue = queue.Queue()
        self._supported = set(supported_extensions)
        self._debounce = debounce_seconds

        self._handler = DebouncedHandler(
            self._queue, self._supported, debounce_seconds
        )
        self._observer = Observer()
        self._worker_thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._observer.schedule(self._handler, str(self._folder), recursive=True)
        self._observer.start()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        self._running = False
        self._observer.stop()
        self._observer.join()

    def scan_folder(self) -> None:
        files = [
            f for f in self._folder.rglob("*")
            if f.is_file() and f.suffix.lower() in self._supported
        ]
        total = len(files)
        for i, f in enumerate(files, 1):
            if self._on_progress:
                self._on_progress(f"Indexing {i}/{total}...")
            self._on_ingest(f)
        if self._on_progress and total > 0:
            self._on_progress("Finished")

    def _worker_loop(self) -> None:
        while self._running:
            try:
                action, path = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if action == "ingest":
                if self._on_progress:
                    self._on_progress(f"Indexing {Path(path).name}...")
                self._on_ingest(Path(path))
            elif action == "delete":
                self._on_delete(Path(path))

            self._queue.task_done()
