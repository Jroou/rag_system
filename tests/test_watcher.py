import queue
import time
from pathlib import Path

import pytest

from src.ingestion.watcher import DebouncedHandler, FileWatcher


class TestDebouncedHandler:
    def test_debounce_delays_processing(self):
        q = queue.Queue()
        handler = DebouncedHandler(q, {".md", ".py"}, debounce_seconds=0.5)

        from unittest.mock import MagicMock

        event = MagicMock()
        event.is_directory = False
        event.src_path = "/tmp/test.md"

        handler.on_created(event)
        assert q.empty()

        time.sleep(1.0)
        assert not q.empty()
        action, path = q.get_nowait()
        assert action == "ingest"
        assert path == "/tmp/test.md"

    def test_delete_bypasses_debounce(self):
        q = queue.Queue()
        handler = DebouncedHandler(q, {".md"}, debounce_seconds=2.0)

        from unittest.mock import MagicMock

        event = MagicMock()
        event.is_directory = False
        event.src_path = "/tmp/deleted.md"

        handler.on_deleted(event)
        assert not q.empty()
        action, path = q.get_nowait()
        assert action == "delete"

    def test_unsupported_extension_ignored(self):
        q = queue.Queue()
        handler = DebouncedHandler(q, {".md"}, debounce_seconds=0.3)

        from unittest.mock import MagicMock

        event = MagicMock()
        event.is_directory = False
        event.src_path = "/tmp/test.jpg"

        handler.on_created(event)
        time.sleep(0.5)
        assert q.empty()


class TestFileWatcher:
    def test_scan_folder(self, tmp_path: Path):
        (tmp_path / "doc1.md").write_text("# Hello")
        (tmp_path / "doc2.py").write_text("x = 1")
        (tmp_path / "ignore.jpg").write_text("binary")

        ingested = []
        deleted = []
        progress = []

        watcher = FileWatcher(
            folder=str(tmp_path),
            supported_extensions=[".md", ".py"],
            on_ingest=lambda p: ingested.append(p),
            on_delete=lambda p: deleted.append(p),
            on_progress=lambda msg: progress.append(msg),
        )
        watcher.scan_folder()

        assert len(ingested) == 2
        assert any("doc1.md" in str(p) for p in ingested)
        assert any("doc2.py" in str(p) for p in ingested)
        assert "Finished" in progress
        assert len(deleted) == 0
