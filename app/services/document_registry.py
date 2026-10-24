"""Durable document identity, generation state, and per-document ingestion locks."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import unicodedata
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def filename_key(filename: str) -> str:
    return unicodedata.normalize("NFKC", Path(filename).name).casefold()


class AmbiguousDocumentError(ValueError):
    def __init__(self, document_ids: list[str]):
        self.document_ids = document_ids
        super().__init__("More than one document has this filename; provide document_id")


class DocumentRegistry:
    """SQLite registry. SQLite is suitable for a single-host deployment.

    Use a shared transactional database with advisory locks for multi-host production.
    """

    LOCK_LEASE_SECONDS = 1800
    LOCK_WAIT_SECONDS = 60

    def __init__(self, path: str):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    source_filename TEXT NOT NULL,
                    filename_key TEXT NOT NULL,
                    active_generation_id TEXT,
                    active_version TEXT,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS documents_filename_idx
                    ON documents(filename_key);
                CREATE TABLE IF NOT EXISTS document_generations (
                    generation_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(document_id),
                    document_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    expected_chunk_ids TEXT NOT NULL DEFAULT '[]',
                    stale_chunk_ids TEXT NOT NULL DEFAULT '[]',
                    warnings TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS generations_document_idx
                    ON document_generations(document_id, created_at);
                CREATE TABLE IF NOT EXISTS document_locks (
                    document_id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    expires_at REAL NOT NULL
                );
                """
            )
            columns = {row["name"] for row in db.execute("PRAGMA table_info(document_generations)")}
            if "stale_chunk_ids" not in columns:
                db.execute(
                    "ALTER TABLE document_generations ADD COLUMN stale_chunk_ids TEXT NOT NULL DEFAULT '[]'"
                )
            document_columns = {row["name"] for row in db.execute("PRAGMA table_info(documents)")}
            if "deleted" not in document_columns:
                db.execute(
                    "ALTER TABLE documents ADD COLUMN deleted INTEGER NOT NULL DEFAULT 0"
                )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout = 10000")
        db.execute("PRAGMA foreign_keys = ON")
        return db

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        db = self._connect()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def resolve_document(
        self,
        source_filename: str,
        *,
        document_id: str | None = None,
        new_document: bool = False,
        legacy_id: str | None = None,
    ) -> str:
        key = filename_key(source_filename)
        now = _now()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if document_id:
                row = db.execute(
                    "SELECT document_id FROM documents WHERE document_id = ?",
                    (document_id,),
                ).fetchone()
                if row is None:
                    raise LookupError("Document ID is not registered")
                db.execute(
                    "UPDATE documents SET source_filename=?, updated_at=? WHERE document_id=?",
                    (Path(source_filename).name, now, document_id),
                )
                return document_id

            if not new_document:
                rows = db.execute(
                    "SELECT document_id FROM documents WHERE filename_key=? ORDER BY created_at",
                    (key,),
                ).fetchall()
                if len(rows) > 1:
                    raise AmbiguousDocumentError([row[0] for row in rows])
                if rows:
                    return str(rows[0][0])

            # First-use adoption keeps the pre-registry filename-hash identity, if
            # present, while making the registry the durable source of truth.
            chosen = legacy_id if legacy_id and not new_document else str(uuid.uuid4())
            db.execute(
                "INSERT INTO documents(document_id, source_filename, filename_key, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (chosen, Path(source_filename).name, key, now, now),
            )
            return chosen

    @contextmanager
    def ingestion_lock(self, document_id: str) -> Iterator[None]:
        owner = str(uuid.uuid4())
        deadline = time.monotonic() + self.LOCK_WAIT_SECONDS
        acquired = False
        while not acquired:
            now = time.time()
            with self._connection() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute("DELETE FROM document_locks WHERE expires_at < ?", (now,))
                try:
                    db.execute(
                        "INSERT INTO document_locks(document_id, owner, expires_at) VALUES (?, ?, ?)",
                        (document_id, owner, now + self.LOCK_LEASE_SECONDS),
                    )
                    acquired = True
                except sqlite3.IntegrityError:
                    acquired = False
            if not acquired:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for document ingestion lock")
                time.sleep(0.1)

        stop = threading.Event()

        def renew() -> None:
            while not stop.wait(self.LOCK_LEASE_SECONDS / 3):
                try:
                    with self._connection() as db:
                        db.execute(
                            "UPDATE document_locks SET expires_at=? WHERE document_id=? AND owner=?",
                            (time.time() + self.LOCK_LEASE_SECONDS, document_id, owner),
                        )
                except sqlite3.Error:
                    pass

        heartbeat = threading.Thread(target=renew, daemon=True)
        heartbeat.start()
        try:
            yield
        finally:
            stop.set()
            heartbeat.join(timeout=1)
            with self._connection() as db:
                db.execute(
                    "DELETE FROM document_locks WHERE document_id=? AND owner=?",
                    (document_id, owner),
                )

    def begin_generation(self, document_id: str, version: str) -> str:
        now = _now()
        with self._connection() as db:
            # An incomplete attempt for the same bytes is resumable. Active
            # generations always get a fresh generation for explicit reprocessing.
            row = db.execute(
                """SELECT generation_id FROM document_generations
                   WHERE document_id=? AND document_version=?
                     AND status IN ('pending','parse_failed','upsert_failed','activation_failed','upserted')
                   ORDER BY created_at DESC LIMIT 1""",
                (document_id, version),
            ).fetchone()
            if row:
                generation_id = str(row[0])
                db.execute(
                    "UPDATE document_generations SET status='pending', updated_at=? WHERE generation_id=?",
                    (now, generation_id),
                )
                return generation_id
            generation_id = str(uuid.uuid4())
            db.execute(
                """INSERT INTO document_generations
                   (generation_id, document_id, document_version, status, created_at, updated_at)
                   VALUES (?, ?, ?, 'pending', ?, ?)""",
                (generation_id, document_id, version, now, now),
            )
            return generation_id

    def set_generation_status(
        self,
        generation_id: str,
        status: str,
        *,
        chunk_ids: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        assignments = ["status=?", "updated_at=?"]
        values: list[object] = [status, _now()]
        if chunk_ids is not None:
            old_row = None
            with self._connection() as db:
                old_row = db.execute(
                    "SELECT expected_chunk_ids, stale_chunk_ids FROM document_generations WHERE generation_id=?",
                    (generation_id,),
                ).fetchone()
            if old_row:
                old_ids = json.loads(old_row["expected_chunk_ids"])
                stale_ids = set(json.loads(old_row["stale_chunk_ids"]))
                stale_ids.update(set(old_ids) - set(chunk_ids))
                assignments.append("stale_chunk_ids=?")
                values.append(json.dumps(sorted(stale_ids)))
            assignments.append("expected_chunk_ids=?")
            values.append(json.dumps(chunk_ids))
        if warnings is not None:
            assignments.append("warnings=?")
            values.append(json.dumps(warnings))
        values.append(generation_id)
        with self._connection() as db:
            db.execute(
                f"UPDATE document_generations SET {', '.join(assignments)} WHERE generation_id=?",
                values,
            )

    def generation_chunk_ids(self, generation_id: str) -> list[str]:
        with self._connection() as db:
            row = db.execute(
                "SELECT expected_chunk_ids FROM document_generations WHERE generation_id=?",
                (generation_id,),
            ).fetchone()
        return json.loads(row[0]) if row else []

    def rotate_generation(self, document_id: str, generation_id: str, version: str) -> str:
        """Abandon a staged attempt if nondeterministic extraction changed its chunks."""
        now = _now()
        fresh_id = str(uuid.uuid4())
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE document_generations SET status='abandoned', updated_at=? "
                "WHERE generation_id=? AND document_id=? AND status<>'active'",
                (now, generation_id, document_id),
            )
            db.execute(
                """INSERT INTO document_generations
                   (generation_id, document_id, document_version, status, created_at, updated_at)
                   VALUES (?, ?, ?, 'pending', ?, ?)""",
                (fresh_id, document_id, version, now, now),
            )
        return fresh_id

    def activate_generation(self, document_id: str, generation_id: str, ids: list[str]) -> None:
        now = _now()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            gen = db.execute(
                "SELECT document_version, status, expected_chunk_ids FROM document_generations "
                "WHERE generation_id=? AND document_id=?",
                (generation_id, document_id),
            ).fetchone()
            if not gen or gen["status"] != "upserted":
                raise RuntimeError("Generation is not verified for activation")
            expected = json.loads(gen["expected_chunk_ids"])
            if len(expected) != len(ids) or set(expected) != set(ids):
                raise RuntimeError("Generation chunk ID validation failed")
            db.execute(
                "UPDATE document_generations SET status='superseded', updated_at=? "
                "WHERE document_id=? AND status='active'",
                (now, document_id),
            )
            db.execute(
                "UPDATE document_generations SET status='active', updated_at=? WHERE generation_id=?",
                (now, generation_id),
            )
            db.execute(
                "UPDATE documents SET active_generation_id=?, active_version=?, deleted=0, updated_at=? WHERE document_id=?",
                (generation_id, gen["document_version"], now, document_id),
            )

    def active_generation_filter(self) -> dict:
        with self._connection() as db:
            active = db.execute(
                "SELECT document_id, active_generation_id, deleted FROM documents"
            ).fetchall()
        active = [row for row in active if row["active_generation_id"] and not row["deleted"]]
        clauses = [
            {"$and": [
                {"document_id": row["document_id"]},
                {"generation_id": row["active_generation_id"]},
            ]}
            for row in active
        ]
        active_ids = [row["document_id"] for row in active]
        with self._connection() as db:
            deleted_ids = [
                row[0] for row in db.execute("SELECT document_id FROM documents WHERE deleted=1")
            ]
        # Continue serving old vectors until a document has an activated new
        # generation. This safely supports vectors predating generation metadata.
        legacy = {"generation_id": {"$exists": False}}
        excluded_legacy_ids = sorted(set(active_ids + deleted_ids))
        if excluded_legacy_ids:
            clauses.append({"$and": [legacy, {"document_id": {"$nin": excluded_legacy_ids}}]})
        else:
            clauses.append(legacy)
        return {"$or": clauses}

    def inactive_generations(self, document_id: str, active_generation_id: str) -> list[str]:
        with self._connection() as db:
            rows = db.execute(
                "SELECT generation_id FROM document_generations WHERE document_id=? AND generation_id<>?",
                (document_id, active_generation_id),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def stale_chunk_ids(self, document_id: str) -> list[str]:
        with self._connection() as db:
            rows = db.execute(
                "SELECT stale_chunk_ids FROM document_generations WHERE document_id=?",
                (document_id,),
            ).fetchall()
        ids: set[str] = set()
        for row in rows:
            ids.update(json.loads(row[0]))
        return sorted(ids)

    def clear_stale_chunk_ids(self, document_id: str) -> None:
        with self._connection() as db:
            db.execute(
                "UPDATE document_generations SET stale_chunk_ids='[]' WHERE document_id=?",
                (document_id,),
            )

    def mark_deleted(self, document_id: str) -> None:
        with self._connection() as db:
            db.execute(
                "UPDATE documents SET active_generation_id=NULL, active_version=NULL, deleted=1, updated_at=? "
                "WHERE document_id=?",
                (_now(), document_id),
            )
            db.execute(
                "UPDATE document_generations SET status='deleted', updated_at=? WHERE document_id=?",
                (_now(), document_id),
            )

