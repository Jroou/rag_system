import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class SQLiteStore:
    def __init__(self, db_path: str):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL UNIQUE,
                document_type TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'indexed',
                indexed_at TEXT NOT NULL,
                error_message TEXT
            );
        """)
        self._conn.commit()

    def get_document_by_path(self, source_path: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE source_path = ?", (source_path,)
        ).fetchone()
        return dict(row) if row else None

    def upsert_document(
        self,
        document_id: str,
        source_path: str,
        document_type: str,
        file_hash: str,
        status: str = "indexed",
        error_message: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO documents (id, source_path, document_type, file_hash, status, indexed_at, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                source_path = excluded.source_path,
                document_type = excluded.document_type,
                file_hash = excluded.file_hash,
                status = excluded.status,
                indexed_at = excluded.indexed_at,
                error_message = excluded.error_message
            """,
            (document_id, source_path, document_type, file_hash, status, now, error_message),
        )
        self._conn.commit()

    def delete_document(self, source_path: str) -> str | None:
        row = self._conn.execute(
            "SELECT id FROM documents WHERE source_path = ?", (source_path,)
        ).fetchone()
        if row:
            doc_id = row["id"]
            self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            self._conn.commit()
            return doc_id
        return None

    def list_documents(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM documents ORDER BY indexed_at DESC").fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()


def compute_file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
