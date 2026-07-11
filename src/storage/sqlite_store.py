import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class SQLiteStore:
    def __init__(self, db_path: str):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
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
                error_message TEXT,
                original_name TEXT,
                thread_id TEXT
            );
            CREATE TABLE IF NOT EXISTS findings (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                citations TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                summary TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS thread_messages (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (thread_id) REFERENCES threads(id)
            );
        """)
        self._conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(documents)").fetchall()}
        if "original_name" not in cols:
            self._conn.execute("ALTER TABLE documents ADD COLUMN original_name TEXT")
            self._conn.commit()
        if "thread_id" not in cols:
            self._conn.execute("ALTER TABLE documents ADD COLUMN thread_id TEXT")
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
        original_name: str | None = None,
        thread_id: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO documents (id, source_path, document_type, file_hash, status, indexed_at, error_message, original_name, thread_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                source_path = excluded.source_path,
                document_type = excluded.document_type,
                file_hash = excluded.file_hash,
                status = excluded.status,
                indexed_at = excluded.indexed_at,
                error_message = excluded.error_message,
                original_name = COALESCE(excluded.original_name, documents.original_name),
                thread_id = excluded.thread_id
            """,
            (document_id, source_path, document_type, file_hash, status, now, error_message, original_name, thread_id),
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

    def get_thread_documents(self, thread_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM documents WHERE thread_id = ?", (thread_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def promote_document_to_global(self, document_id: str) -> None:
        self._conn.execute(
            "UPDATE documents SET thread_id = NULL WHERE id = ?", (document_id,)
        )
        self._conn.commit()

    def add_finding(self, finding_id: str, text: str, citations: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO findings (id, text, citations, created_at) VALUES (?, ?, ?, ?)",
            (finding_id, text, citations, now),
        )
        self._conn.commit()

    def list_findings(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM findings ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_finding(self, finding_id: str) -> None:
        self._conn.execute("DELETE FROM findings WHERE id = ?", (finding_id,))
        self._conn.commit()

    # --- Thread persistence ---

    def create_thread(self, thread_id: str, title: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO threads (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (thread_id, title, now, now),
        )
        self._conn.commit()

    def update_thread_title(self, thread_id: str, title: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE threads SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, thread_id),
        )
        self._conn.commit()

    def update_thread_summary(self, thread_id: str, summary: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE threads SET summary = ?, updated_at = ? WHERE id = ?",
            (summary, now, thread_id),
        )
        self._conn.commit()

    def touch_thread(self, thread_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE threads SET updated_at = ? WHERE id = ?", (now, thread_id)
        )
        self._conn.commit()

    def list_threads(self, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM threads ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_thread(self, thread_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM threads WHERE id = ?", (thread_id,)
        ).fetchone()
        return dict(row) if row else None

    def add_thread_message(self, message_id: str, thread_id: str, role: str, content: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO thread_messages (id, thread_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (message_id, thread_id, role, content, now),
        )
        self._conn.commit()
        self.touch_thread(thread_id)

    def get_thread_messages(self, thread_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM thread_messages WHERE thread_id = ? ORDER BY created_at ASC",
            (thread_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_thread(self, thread_id: str) -> None:
        self._conn.execute("DELETE FROM thread_messages WHERE thread_id = ?", (thread_id,))
        self._conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def compute_file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
