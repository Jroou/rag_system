from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional

import chainlit as cl
from chainlit.data.base import BaseDataLayer
from chainlit.data.utils import queue_until_user_message
from chainlit.types import (
    Feedback,
    PageInfo,
    Pagination,
    PaginatedResponse,
    ThreadFilter,
)
from chainlit.user import PersistedUser, User

if TYPE_CHECKING:
    from chainlit.element import Element
    from chainlit.step import StepDict
    from chainlit.types import ElementDict, FeedbackDict, ThreadDict

from src.storage.sqlite_store import SQLiteStore

_LOCAL_USER_ID = "local"
_LOCAL_USER_IDENTIFIER = "local"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteDataLayer(BaseDataLayer):
    """Chainlit BaseDataLayer backed by the app's existing SQLiteStore."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store
        self._ensure_steps_table()

    def _ensure_steps_table(self) -> None:
        self._store._conn.executescript("""
            CREATE TABLE IF NOT EXISTS steps (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                type TEXT NOT NULL,
                name TEXT,
                input TEXT,
                output TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                start TEXT,
                end TEXT,
                parent_id TEXT
            );
            CREATE TABLE IF NOT EXISTS elements (
                id TEXT PRIMARY KEY,
                thread_id TEXT,
                type TEXT,
                name TEXT,
                url TEXT,
                object_key TEXT,
                display TEXT,
                mime TEXT,
                created_at TEXT NOT NULL
            );
        """)
        self._store._conn.commit()

    # ------------------------------------------------------------------
    # User
    # ------------------------------------------------------------------

    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        return PersistedUser(
            id=_LOCAL_USER_ID,
            identifier=_LOCAL_USER_IDENTIFIER,
            createdAt=_now(),
            metadata={},
        )

    async def create_user(self, user: User) -> Optional[PersistedUser]:
        return PersistedUser(
            id=_LOCAL_USER_ID,
            identifier=_LOCAL_USER_IDENTIFIER,
            createdAt=_now(),
            metadata={},
        )

    # ------------------------------------------------------------------
    # Feedback (no-op — not used in this app)
    # ------------------------------------------------------------------

    async def delete_feedback(self, feedback_id: str) -> bool:
        return True

    async def upsert_feedback(self, feedback: Feedback) -> str:
        return feedback.id or ""

    # ------------------------------------------------------------------
    # Elements
    # ------------------------------------------------------------------

    @queue_until_user_message()
    async def create_element(self, element: "Element") -> None:
        self._store._conn.execute(
            """INSERT OR REPLACE INTO elements
               (id, thread_id, type, name, url, object_key, display, mime, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                element.id,
                element.thread_id,
                element.type,
                element.name,
                getattr(element, "url", None),
                getattr(element, "object_key", None),
                element.display,
                getattr(element, "mime", None),
                _now(),
            ),
        )
        self._store._conn.commit()

    async def get_element(self, thread_id: str, element_id: str) -> Optional["ElementDict"]:
        row = self._store._conn.execute(
            "SELECT * FROM elements WHERE id = ? AND thread_id = ?",
            (element_id, thread_id),
        ).fetchone()
        if not row:
            return None
        r = dict(row)
        return {
            "id": r["id"],
            "threadId": r["thread_id"],
            "type": r["type"],
            "name": r["name"],
            "url": r["url"],
            "objectKey": r["object_key"],
            "display": r["display"],
            "mime": r["mime"],
        }

    @queue_until_user_message()
    async def delete_element(self, element_id: str, thread_id: Optional[str] = None) -> None:
        self._store._conn.execute("DELETE FROM elements WHERE id = ?", (element_id,))
        self._store._conn.commit()

    # ------------------------------------------------------------------
    # Steps (map to thread_messages for user/assistant, full steps table for rest)
    # ------------------------------------------------------------------

    @queue_until_user_message()
    async def create_step(self, step_dict: "StepDict") -> None:
        thread_id = step_dict.get("threadId", "")
        step_type = step_dict.get("type", "")
        # Ensure thread exists (Chainlit calls update_thread before create_step,
        # but guard anyway)
        await self.update_thread(thread_id)

        self._store._conn.execute(
            """INSERT OR REPLACE INTO steps
               (id, thread_id, type, name, input, output, metadata, created_at, start, end, parent_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                step_dict.get("id", ""),
                thread_id,
                step_type,
                step_dict.get("name", ""),
                step_dict.get("input", ""),
                step_dict.get("output", ""),
                "{}",
                step_dict.get("createdAt") or _now(),
                step_dict.get("start"),
                step_dict.get("end"),
                step_dict.get("parentId"),
            ),
        )
        self._store._conn.commit()

        # Mirror user/assistant messages to thread_messages for RAG history
        if step_type == "user_message":
            content = step_dict.get("output") or step_dict.get("input", "")
            if content:
                self._store.add_thread_message(step_dict.get("id", ""), thread_id, "user", content)
        elif step_type == "assistant_message":
            content = step_dict.get("output", "")
            if content:
                self._store.add_thread_message(step_dict.get("id", ""), thread_id, "assistant", content)

    @queue_until_user_message()
    async def update_step(self, step_dict: "StepDict") -> None:
        self._store._conn.execute(
            """UPDATE steps SET
               name = ?, input = ?, output = ?, end = ?
               WHERE id = ?""",
            (
                step_dict.get("name", ""),
                step_dict.get("input", ""),
                step_dict.get("output", ""),
                step_dict.get("end"),
                step_dict.get("id", ""),
            ),
        )
        self._store._conn.commit()

        # Update mirrored message for assistant steps
        step_type = step_dict.get("type", "")
        if step_type == "assistant_message":
            content = step_dict.get("output", "")
            step_id = step_dict.get("id", "")
            if content and step_id:
                existing = self._store._conn.execute(
                    "SELECT id FROM thread_messages WHERE id = ?", (step_id,)
                ).fetchone()
                if existing:
                    self._store._conn.execute(
                        "UPDATE thread_messages SET content = ? WHERE id = ?",
                        (content, step_id),
                    )
                    self._store._conn.commit()

    @queue_until_user_message()
    async def delete_step(self, step_id: str) -> None:
        self._store._conn.execute("DELETE FROM steps WHERE id = ?", (step_id,))
        self._store._conn.execute("DELETE FROM thread_messages WHERE id = ?", (step_id,))
        self._store._conn.commit()

    # ------------------------------------------------------------------
    # Threads
    # ------------------------------------------------------------------

    async def get_thread_author(self, thread_id: str) -> str:
        return _LOCAL_USER_IDENTIFIER

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        now = _now()
        existing = self._store._conn.execute(
            "SELECT id FROM threads WHERE id = ?", (thread_id,)
        ).fetchone()
        if existing:
            if name is not None:
                self._store._conn.execute(
                    "UPDATE threads SET title = ?, updated_at = ? WHERE id = ?",
                    (name, now, thread_id),
                )
        else:
            self._store._conn.execute(
                "INSERT INTO threads (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (thread_id, name or "New conversation", now, now),
            )
        self._store._conn.commit()

    async def get_thread(self, thread_id: str) -> Optional["ThreadDict"]:
        row = self._store._conn.execute(
            "SELECT * FROM threads WHERE id = ?", (thread_id,)
        ).fetchone()
        if not row:
            return None
        thread = dict(row)

        step_rows = self._store._conn.execute(
            "SELECT * FROM steps WHERE thread_id = ? ORDER BY created_at ASC",
            (thread_id,),
        ).fetchall()
        steps: List["StepDict"] = [
            {
                "id": r["id"],
                "threadId": thread_id,
                "type": r["type"],
                "name": r["name"] or "",
                "input": r["input"] or "",
                "output": r["output"] or "",
                "createdAt": r["created_at"],
                "start": r["start"],
                "end": r["end"],
                "parentId": r["parent_id"],
                "metadata": {},
            }
            for r in step_rows
        ]

        element_rows = self._store._conn.execute(
            "SELECT * FROM elements WHERE thread_id = ?", (thread_id,)
        ).fetchall()
        elements = [
            {
                "id": r["id"],
                "threadId": r["thread_id"],
                "type": r["type"],
                "name": r["name"],
                "url": r["url"],
                "display": r["display"],
            }
            for r in element_rows
        ]

        return {
            "id": thread["id"],
            "createdAt": thread["created_at"],
            "name": thread["title"] or "New conversation",
            "userId": _LOCAL_USER_ID,
            "userIdentifier": _LOCAL_USER_IDENTIFIER,
            "tags": [],
            "metadata": {},
            "steps": steps,
            "elements": elements,
        }

    async def list_threads(
        self,
        pagination: Pagination,
        filters: ThreadFilter,
    ) -> "PaginatedResponse[ThreadDict]":
        limit = pagination.first
        cursor = pagination.cursor
        search = filters.search or ""

        if cursor:
            cursor_row = self._store._conn.execute(
                "SELECT updated_at FROM threads WHERE id = ?", (cursor,)
            ).fetchone()
            cursor_ts = cursor_row["updated_at"] if cursor_row else ""
        else:
            cursor_ts = ""

        if search:
            query = """
                SELECT * FROM threads
                WHERE (? = '' OR updated_at < ?)
                AND title LIKE ?
                ORDER BY updated_at DESC LIMIT ?
            """
            rows = self._store._conn.execute(
                query, (cursor_ts, cursor_ts, f"%{search}%", limit + 1)
            ).fetchall()
        else:
            query = """
                SELECT * FROM threads
                WHERE (? = '' OR updated_at < ?)
                ORDER BY updated_at DESC LIMIT ?
            """
            rows = self._store._conn.execute(
                query, (cursor_ts, cursor_ts, limit + 1)
            ).fetchall()

        has_next = len(rows) > limit
        rows = rows[:limit]

        threads: List["ThreadDict"] = [
            {
                "id": r["id"],
                "createdAt": r["created_at"],
                "name": r["title"] or "New conversation",
                "userId": _LOCAL_USER_ID,
                "userIdentifier": _LOCAL_USER_IDENTIFIER,
                "tags": [],
                "metadata": {},
                "steps": [],
                "elements": [],
            }
            for r in rows
        ]

        return PaginatedResponse(
            pageInfo=PageInfo(
                hasNextPage=has_next,
                startCursor=threads[0]["id"] if threads else None,
                endCursor=threads[-1]["id"] if threads else None,
            ),
            data=threads,
        )

    async def delete_thread(self, thread_id: str) -> None:
        self._store._conn.execute("DELETE FROM elements WHERE thread_id = ?", (thread_id,))
        self._store._conn.execute("DELETE FROM steps WHERE thread_id = ?", (thread_id,))
        self._store._conn.execute("DELETE FROM thread_messages WHERE thread_id = ?", (thread_id,))
        self._store._conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
        self._store._conn.commit()

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    async def build_debug_url(self) -> str:
        return ""

    async def close(self) -> None:
        pass

    async def get_favorite_steps(self, user_id: str) -> List["StepDict"]:
        return []
