"""Conversation ownership records, separate from LangGraph checkpoints and audit data."""

import hashlib
import secrets
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def open_conversation_store(path: str) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path), check_same_thread=False)
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS conversations (
            thread_id TEXT PRIMARY KEY,
            owner_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )"""
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS conversations_owner_idx ON conversations(owner_hash)"
    )
    connection.commit()
    return connection


def new_owner_token() -> str:
    return secrets.token_urlsafe(32)


def owner_hash(owner_token: str) -> str:
    return hashlib.sha256(owner_token.encode("utf-8")).hexdigest()


def create_conversation(connection: sqlite3.Connection, owner_token: str) -> str:
    thread_id = str(uuid.uuid4())
    connection.execute(
        "INSERT INTO conversations(thread_id, owner_hash, created_at) VALUES (?, ?, ?)",
        (thread_id, owner_hash(owner_token), datetime.now(timezone.utc).isoformat()),
    )
    connection.commit()
    return thread_id


def owns_conversation(
    connection: sqlite3.Connection, thread_id: str, owner_token: str
) -> bool:
    row = connection.execute(
        "SELECT 1 FROM conversations WHERE thread_id = ? AND owner_hash = ?",
        (thread_id, owner_hash(owner_token)),
    ).fetchone()
    return row is not None


def read_messages(graph, thread_id: str) -> list[dict[str, str]]:
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    result: list[dict[str, str]] = []
    for message in snapshot.values.get("messages", []):
        content = message.content
        if not isinstance(content, str) or not content.strip():
            continue
        message_type = getattr(message, "type", "")
        if message_type in ("human", "ai"):
            result.append({"role": "user" if message_type == "human" else "assistant", "content": content})
    return result
