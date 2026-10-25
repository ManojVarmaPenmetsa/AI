"""SQLite audit records, deliberately separate from conversation/checkpoint storage."""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import get_settings


settings = get_settings()
logger = logging.getLogger(__name__)

_QUERY_COLUMNS = {
    "request_id": "TEXT",
    "conversation_id": "TEXT",
    "outcome": "TEXT",
    "selected_route": "TEXT",
    "retrieval_json": "TEXT NOT NULL DEFAULT '[]'",
    "evidence_json": "TEXT NOT NULL DEFAULT '{}'",
    "web_fallback": "INTEGER NOT NULL DEFAULT 0",
    "query_rewrite_count": "INTEGER NOT NULL DEFAULT 0",
    "retry_count": "INTEGER NOT NULL DEFAULT 0",
    "latency_ms": "REAL",
    "stage_timings_json": "TEXT NOT NULL DEFAULT '{}'",
    "error_category": "TEXT",
    "diagnostics_json": "TEXT NOT NULL DEFAULT '[]'",
    "question_chars": "INTEGER",
}
_INGESTION_COLUMNS = {
    "request_id": "TEXT",
    "latency_ms": "REAL",
    "error_category": "TEXT",
}


def _connect() -> sqlite3.Connection:
    path = Path(settings.audit_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


@contextmanager
def _connection():
    connection = _connect()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _migrate_columns(connection: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
    for name, declaration in columns.items():
        if name not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")


def init_db() -> None:
    """Create current schema and add fields/indexes to existing audit databases."""
    with _connection() as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS query_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                question TEXT NOT NULL,
                source_used TEXT NOT NULL,
                trace_json TEXT NOT NULL
            )"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS ingestion_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                document_id TEXT NOT NULL,
                source_filename TEXT NOT NULL,
                document_version TEXT NOT NULL,
                status TEXT NOT NULL,
                vision_status TEXT NOT NULL,
                diagnostics_json TEXT NOT NULL
            )"""
        )
        _migrate_columns(connection, "query_audit", _QUERY_COLUMNS)
        _migrate_columns(connection, "ingestion_audit", _INGESTION_COLUMNS)
        for name, column in (
            ("idx_query_created_at", "created_at"),
            ("idx_query_request_id", "request_id"),
            ("idx_query_conversation_id", "conversation_id"),
            ("idx_query_outcome", "outcome"),
            ("idx_query_selected_route", "selected_route"),
            ("idx_ingestion_created_at", "created_at"),
            ("idx_ingestion_request_id", "request_id"),
            ("idx_ingestion_document_id", "document_id"),
            ("idx_ingestion_status", "status"),
        ):
            table = "ingestion_audit" if name.startswith("idx_ingestion") else "query_audit"
            connection.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table}({column})")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def write_audit(
    question: str,
    source_used: str,
    trace: list[str],
    *,
    request_id: str | None = None,
    conversation_id: str | None = None,
    outcome: str | None = None,
    selected_route: str | None = None,
    retrieval_results: list[dict[str, Any]] | None = None,
    evidence: dict[str, Any] | None = None,
    web_fallback: bool = False,
    query_rewrite_count: int = 0,
    retry_count: int = 0,
    latency_ms: float | None = None,
    stage_timings_ms: dict[str, float] | None = None,
    error_category: str | None = None,
    diagnostics: list[str] | None = None,
) -> int:
    """Persist metadata only; new records do not retain the user's query text."""
    if outcome is None:
        outcome = "insufficient_evidence" if source_used == "insufficient_evidence" else "answered"
    with _connection() as connection:
        cursor = connection.execute(
            """INSERT INTO query_audit(
                created_at, question, source_used, trace_json, request_id,
                conversation_id, outcome, selected_route, retrieval_json,
                evidence_json, web_fallback, query_rewrite_count, retry_count,
                latency_ms, stage_timings_json, error_category, diagnostics_json,
                question_chars
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(), "", source_used, _json(trace),
                request_id, conversation_id, outcome, selected_route,
                _json(retrieval_results or []), _json(evidence or {}), int(web_fallback),
                max(0, int(query_rewrite_count)), max(0, int(retry_count)), latency_ms,
                _json(stage_timings_ms or {}), error_category, _json(diagnostics or []),
                len(question) if question else 0,
            ),
        )
        return int(cursor.lastrowid)


def write_ingestion_audit(
    document_id: str,
    source_filename: str,
    document_version: str,
    status: str,
    vision_status: str,
    diagnostics: list[str],
    *,
    request_id: str | None = None,
    latency_ms: float | None = None,
    error_category: str | None = None,
) -> int:
    """Persist ingestion outcome and safe stage diagnostics, never document contents."""
    with _connection() as connection:
        cursor = connection.execute(
            """INSERT INTO ingestion_audit(
                created_at, document_id, source_filename, document_version,
                status, vision_status, diagnostics_json, request_id, latency_ms,
                error_category
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(), document_id, source_filename,
                document_version, status, vision_status, _json(diagnostics), request_id,
                latency_ms, error_category,
            ),
        )
        return int(cursor.lastrowid)


def _decode(raw: str | None, fallback: Any) -> Any:
    try:
        return json.loads(raw) if raw else fallback
    except (TypeError, json.JSONDecodeError):
        return fallback


def _safe_trace(raw: str | None) -> list[str]:
    trace = _decode(raw, [])
    if not isinstance(trace, list):
        return []
    return ["Query rewrite" if isinstance(item, str) and item.startswith("Query rewrite") else str(item) for item in trace]


def _audit_filter_sql(
    *, conversation_id: str | None = None, request_id: str | None = None,
    outcome: str | None = None, selected_route: str | None = None,
    from_ts: str | None = None, to_ts: str | None = None,
) -> tuple[str, list[str]]:
    clauses: list[str] = []
    values: list[str] = []
    for column, value in (
        ("conversation_id", conversation_id), ("request_id", request_id),
        ("selected_route", selected_route),
    ):
        if value:
            clauses.append(f"{column} = ?")
            values.append(value)
    if outcome:
        clauses.append("COALESCE(outcome, CASE WHEN source_used='insufficient_evidence' THEN 'insufficient_evidence' ELSE 'answered' END) = ?")
        values.append(outcome)
    for column, value, operator in (("created_at", from_ts, ">="), ("created_at", to_ts, "<=")):
        if value:
            clauses.append(f"{column} {operator} ?")
            values.append(value)
    return (" WHERE " + " AND ".join(clauses) if clauses else "", values)


def list_query_audit(
    *, limit: int = 50, offset: int = 0, conversation_id: str | None = None,
    request_id: str | None = None, outcome: str | None = None,
    selected_route: str | None = None,
    from_ts: str | None = None, to_ts: str | None = None,
) -> dict[str, Any]:
    where, values = _audit_filter_sql(
        conversation_id=conversation_id, request_id=request_id, outcome=outcome,
        selected_route=selected_route, from_ts=from_ts, to_ts=to_ts,
    )
    with _connection() as connection:
        total = connection.execute(f"SELECT COUNT(*) FROM query_audit{where}", values).fetchone()[0]
        rows = connection.execute(
            f"""SELECT id, created_at, source_used, trace_json, request_id,
                conversation_id, outcome, selected_route, retrieval_json, evidence_json,
                web_fallback, query_rewrite_count, retry_count, latency_ms,
                stage_timings_json, error_category, diagnostics_json,
                question_chars FROM query_audit{where}
                ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?""",
            [*values, limit, offset],
        ).fetchall()
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [
            {
                "id": row["id"], "created_at": row["created_at"],
                "request_id": row["request_id"], "conversation_id": row["conversation_id"],
                "outcome": row["outcome"] or ("insufficient_evidence" if row["source_used"] == "insufficient_evidence" else "answered"),
                "source_used": row["source_used"], "selected_route": row["selected_route"],
                "retrieval_results": _decode(row["retrieval_json"], []),
                "evidence": _decode(row["evidence_json"], {}),
                "web_fallback": bool(row["web_fallback"]),
                "query_rewrite_count": row["query_rewrite_count"],
                "retry_count": row["retry_count"], "latency_ms": row["latency_ms"],
                "stage_timings_ms": _decode(row["stage_timings_json"], {}),
                "error_category": row["error_category"],
                "diagnostics": _decode(row["diagnostics_json"], []),
                "question_chars": row["question_chars"],
                "trace": _safe_trace(row["trace_json"]),
            }
            for row in rows
        ],
    }


def audit_metrics(
    *, from_ts: str | None = None, to_ts: str | None = None,
    outcome: str | None = None, selected_route: str | None = None,
) -> dict[str, Any]:
    where, values = _audit_filter_sql(
        from_ts=from_ts, to_ts=to_ts, outcome=outcome, selected_route=selected_route,
    )
    ingestion_clauses: list[str] = []
    ingestion_values: list[str] = []
    if from_ts:
        ingestion_clauses.append("created_at >= ?")
        ingestion_values.append(from_ts)
    if to_ts:
        ingestion_clauses.append("created_at <= ?")
        ingestion_values.append(to_ts)
    ingestion_where = " WHERE " + " AND ".join(ingestion_clauses) if ingestion_clauses else ""
    with _connection() as connection:
        query = connection.execute(
            f"""SELECT COUNT(*) AS total,
                SUM(CASE WHEN COALESCE(outcome, CASE WHEN source_used='insufficient_evidence' THEN 'insufficient_evidence' ELSE 'answered' END)='answered' THEN 1 ELSE 0 END) AS answered,
                SUM(CASE WHEN COALESCE(outcome, CASE WHEN source_used='insufficient_evidence' THEN 'insufficient_evidence' ELSE 'answered' END)='insufficient_evidence' THEN 1 ELSE 0 END) AS insufficient_evidence,
                SUM(CASE WHEN outcome='failed' THEN 1 ELSE 0 END) AS failed,
                SUM(CASE WHEN web_fallback=1 THEN 1 ELSE 0 END) AS web_fallback_requests,
                SUM(CASE WHEN source_used='private_kb' THEN 1 ELSE 0 END) AS private_kb_requests,
                AVG(latency_ms) AS average_latency_ms
                FROM query_audit{where}""", values,
        ).fetchone()
        ingestion = connection.execute(
            f"SELECT status, COUNT(*) AS count FROM ingestion_audit{ingestion_where} GROUP BY status",
            ingestion_values,
        ).fetchall()
    return {
        "query_requests": {key: query[key] or 0 for key in (
            "total", "answered", "insufficient_evidence", "failed", "web_fallback_requests"
        )} | {
            "private_kb_requests": query["private_kb_requests"] or 0,
            "failure_rate": (query["failed"] or 0) / query["total"] if query["total"] else None,
            "average_latency_ms": query["average_latency_ms"],
        },
        "ingestion_by_status": {row["status"]: row["count"] for row in ingestion},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def list_ingestion_audit(
    *, limit: int = 50, offset: int = 0, document_id: str | None = None,
    request_id: str | None = None, status: str | None = None,
    from_ts: str | None = None, to_ts: str | None = None,
) -> dict[str, Any]:
    clauses: list[str] = []
    values: list[str] = []
    for column, value in (("document_id", document_id), ("request_id", request_id), ("status", status)):
        if value:
            clauses.append(f"{column} = ?")
            values.append(value)
    if from_ts:
        clauses.append("created_at >= ?")
        values.append(from_ts)
    if to_ts:
        clauses.append("created_at <= ?")
        values.append(to_ts)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with _connection() as connection:
        total = connection.execute(f"SELECT COUNT(*) FROM ingestion_audit{where}", values).fetchone()[0]
        rows = connection.execute(
            f"""SELECT id, created_at, request_id, document_id, source_filename,
                document_version, status, vision_status, diagnostics_json, latency_ms,
                error_category FROM ingestion_audit{where}
                ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?""",
            [*values, limit, offset],
        ).fetchall()
    return {
        "total": total, "limit": limit, "offset": offset,
        "items": [
            {
                "id": row["id"], "created_at": row["created_at"],
                "request_id": row["request_id"], "document_id": row["document_id"],
                "source_filename": row["source_filename"], "document_version": row["document_version"],
                "status": row["status"], "vision_status": row["vision_status"],
                "diagnostics": _decode(row["diagnostics_json"], []),
                "latency_ms": row["latency_ms"], "error_category": row["error_category"],
            }
            for row in rows
        ],
    }
