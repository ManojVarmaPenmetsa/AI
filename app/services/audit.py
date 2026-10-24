import json
import sqlite3
from datetime import datetime, timezone
from app.core.config import get_settings

settings = get_settings()

def init_db() -> None:
    con = sqlite3.connect(settings.audit_db_path)
    con.execute(
        """CREATE TABLE IF NOT EXISTS query_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            question TEXT NOT NULL,
            source_used TEXT NOT NULL,
            trace_json TEXT NOT NULL
        )"""
    )
    con.execute(
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
    con.commit()
    con.close()

def write_audit(question: str, source_used: str, trace: list[str]) -> None:
    con = sqlite3.connect(settings.audit_db_path)
    con.execute(
        "INSERT INTO query_audit(created_at, question, source_used, trace_json) VALUES (?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), question, source_used, json.dumps(trace)),
    )
    con.commit()
    con.close()


def write_ingestion_audit(
    document_id: str,
    source_filename: str,
    document_version: str,
    status: str,
    vision_status: str,
    diagnostics: list[str],
) -> None:
    """Persist ingestion state and safe warnings, never document contents or secrets."""
    con = sqlite3.connect(settings.audit_db_path)
    con.execute(
        """INSERT INTO ingestion_audit(
            created_at, document_id, source_filename, document_version,
            status, vision_status, diagnostics_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            document_id,
            source_filename,
            document_version,
            status,
            vision_status,
            json.dumps(diagnostics),
        ),
    )
    con.commit()
    con.close()
