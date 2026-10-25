from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main as app_main
from app.core.config import get_settings
from app.services import audit
from app.api.routes import _safe_trace, _safe_web_references, _write_query_audit_safely


class AuditDatabaseTests(unittest.TestCase):
    def test_audit_write_failure_is_swallowed_for_chat_and_does_not_expose_error(self):
        with patch("app.api.routes.write_audit", side_effect=RuntimeError("sensitive database detail")):
            _write_query_audit_safely("question", "failed", [], request_id="req-1")

    def test_audit_sanitizers_remove_rewritten_queries_and_url_parameters(self):
        self.assertEqual(_safe_trace(["Query rewrite → sensitive search text", "KB evidence grade → GOOD"]), ["Query rewrite", "KB evidence grade → GOOD"])
        self.assertEqual(_safe_web_references([
            {"type": "web", "title": "Docs", "url": "https://example.com/path?token=secret#section"},
            {"type": "web", "title": "Bad", "url": "javascript:alert(1)"},
        ]), [{"title": "Docs", "url": "https://example.com/path", "type": "web"}])

    def test_migration_preserves_legacy_rows_and_adds_filter_indexes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit.db"
            connection = sqlite3.connect(path)
            connection.execute("CREATE TABLE query_audit (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, question TEXT NOT NULL, source_used TEXT NOT NULL, trace_json TEXT NOT NULL)")
            connection.execute("INSERT INTO query_audit(created_at, question, source_used, trace_json) VALUES ('2026-01-01T00:00:00+00:00', 'old question', 'insufficient_evidence', '[]')")
            connection.execute("CREATE TABLE ingestion_audit (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, document_id TEXT NOT NULL, source_filename TEXT NOT NULL, document_version TEXT NOT NULL, status TEXT NOT NULL, vision_status TEXT NOT NULL, diagnostics_json TEXT NOT NULL)")
            connection.commit()
            connection.close()
            with patch.object(audit, "settings", SimpleNamespace(audit_db_path=str(path))):
                audit.init_db()
                audit.write_audit(
                    "sensitive question", "private_kb", ["Query rewrite"],
                    request_id="req-1", conversation_id="conv-1", selected_route="kb",
                    retrieval_results=[{"source": "guide.pdf", "page": 2, "chunk_id": "chunk-1"}],
                    evidence={"kb_grade": "good"}, latency_ms=21.5,
                    stage_timings_ms={"retrieve_kb": 4.2},
                )
                audit.write_audit("unknown policy", "insufficient_evidence", [], request_id="req-2")
                audit.write_ingestion_audit(
                    "doc-1", "guide.pdf", "version-1", "failed", "partial", ["stage:upsert"],
                    request_id="req-upload", latency_ms=50, error_category="upsert",
                )
                listed = audit.list_query_audit(conversation_id="conv-1")
                metrics = audit.audit_metrics()
                ingestion = audit.list_ingestion_audit(status="failed")
            connection = sqlite3.connect(path)
            try:
                legacy = connection.execute("SELECT question, source_used FROM query_audit WHERE id=1").fetchone()
                new_question = connection.execute("SELECT question, question_chars FROM query_audit WHERE request_id='req-1'").fetchone()
                indexes = {row[1] for row in connection.execute("PRAGMA index_list(query_audit)")}
            finally:
                connection.close()

        self.assertEqual(legacy, ("old question", "insufficient_evidence"))
        self.assertEqual(new_question, ("", len("sensitive question")))
        self.assertIn("idx_query_conversation_id", indexes)
        self.assertEqual(listed["total"], 1)
        self.assertEqual(listed["items"][0]["outcome"], "answered")
        self.assertEqual(listed["items"][0]["retrieval_results"][0]["page"], 2)
        self.assertEqual(metrics["query_requests"]["insufficient_evidence"], 2)
        self.assertEqual(metrics["ingestion_by_status"], {"failed": 1})
        self.assertEqual(ingestion["items"][0]["error_category"], "upsert")

    def test_admin_audit_endpoints_require_admin_key_and_return_request_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = get_settings().model_copy(update={
                "audit_db_path": str(Path(temp_dir) / "audit.db"),
                "conversation_db_path": str(Path(temp_dir) / "conversations.db"),
                "document_registry_db_path": str(Path(temp_dir) / "registry.db"),
                "admin_api_key": "test-admin-key",
            })
            with (
                patch.object(app_main, "get_settings", return_value=settings),
                patch.object(audit, "settings", settings),
                patch("app.api.routes.settings", settings),
            ):
                with TestClient(app_main.create_app()) as client:
                    denied = client.get("/api/admin/audit/metrics")
                    self.assertEqual(denied.status_code, 401)
                    response = client.get(
                        "/api/admin/audit/metrics",
                        headers={"X-Admin-Key": "test-admin-key"},
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertTrue(response.headers.get("X-Request-ID"))
                    self.assertEqual(response.json()["query_requests"]["total"], 0)


if __name__ == "__main__":
    unittest.main()
