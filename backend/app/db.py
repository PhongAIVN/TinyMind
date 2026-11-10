"""Chat history in SQLite (backend/data/tinymind.db). One connection per call: simple and thread-safe."""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    thinking TEXT NOT NULL DEFAULT '',
    tool_calls TEXT NOT NULL DEFAULT '[]',
    model TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""
MAX_MEMORIES = 30


class ChatStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def create_conversation(self, title: str, model: str) -> dict:
        now = time.time()
        row = {"id": uuid.uuid4().hex, "title": title, "model": model, "created_at": now, "updated_at": now}
        with self._connect() as db:
            db.execute("INSERT INTO conversations VALUES (:id, :title, :model, :created_at, :updated_at)", row)
        return row

    def list_conversations(self) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM conversations ORDER BY updated_at DESC").fetchall()
        return [dict(row) for row in rows]

    def get_conversation(self, conversation_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
            if row is None:
                return None
            messages = db.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,)
            ).fetchall()
        return {**dict(row), "messages": [_message(m) for m in messages]}

    def add_message(self, conversation_id: str, role: str, content: str, *, thinking: str = "",
                    tool_calls: list[dict] | None = None, model: str | None = None) -> dict:
        now = time.time()
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO messages (conversation_id, role, content, thinking, tool_calls, model, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (conversation_id, role, content, thinking, json.dumps(tool_calls or [], ensure_ascii=False),
                 model, now),
            )
            db.execute("UPDATE conversations SET updated_at = ?, model = COALESCE(?, model) WHERE id = ?",
                       (now, model, conversation_id))
            row = db.execute("SELECT * FROM messages WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return _message(row)

    def rename(self, conversation_id: str, title: str) -> bool:
        with self._connect() as db:
            return db.execute("UPDATE conversations SET title = ? WHERE id = ?",
                              (title, conversation_id)).rowcount > 0

    def delete(self, conversation_id: str) -> bool:
        with self._connect() as db:
            return db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,)).rowcount > 0

    def delete_all(self) -> int:
        with self._connect() as db:
            return db.execute("DELETE FROM conversations").rowcount


    def remember(self, kind: str, value: str, single_valued: bool) -> dict:
        """Store a fact. Single-valued kinds replace the old value; exact duplicates are skipped.
        Keeps only the newest MAX_MEMORIES rows."""
        with self._connect() as db:
            if single_valued:
                db.execute("DELETE FROM memories WHERE kind = ?", (kind,))
            else:
                db.execute("DELETE FROM memories WHERE kind = ? AND lower(value) = lower(?)", (kind, value))
            cursor = db.execute("INSERT INTO memories (kind, value, created_at) VALUES (?, ?, ?)",
                                (kind, value, time.time()))
            db.execute("DELETE FROM memories WHERE id NOT IN (SELECT id FROM memories ORDER BY id DESC LIMIT ?)",
                       (MAX_MEMORIES,))
            row = db.execute("SELECT * FROM memories WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return dict(row)

    def list_memories(self) -> list[dict]:
        with self._connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM memories ORDER BY id").fetchall()]

    def forget(self, memory_id: int) -> bool:
        with self._connect() as db:
            return db.execute("DELETE FROM memories WHERE id = ?", (memory_id,)).rowcount > 0

    def forget_all(self) -> int:
        with self._connect() as db:
            return db.execute("DELETE FROM memories").rowcount


def _message(row: sqlite3.Row) -> dict:
    return {**dict(row), "tool_calls": json.loads(row["tool_calls"])}
