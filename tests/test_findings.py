import tempfile
from pathlib import Path

from src.storage.sqlite_store import SQLiteStore


def test_findings_crud():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteStore(db_path=str(Path(tmpdir) / "test.db"))

        assert store.list_findings() == []

        store.add_finding("f1", "The sky is blue [source.pdf]", '[{"source": "source.pdf", "text": "sky"}]')
        store.add_finding("f2", "Water is wet [doc.md]", '[{"source": "doc.md", "text": "water"}]')

        findings = store.list_findings()
        assert len(findings) == 2
        assert findings[0]["id"] == "f2"  # most recent first
        assert findings[1]["id"] == "f1"

        store.delete_finding("f1")
        findings = store.list_findings()
        assert len(findings) == 1
        assert findings[0]["id"] == "f2"

        store.close()
