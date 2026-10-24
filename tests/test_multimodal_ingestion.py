from __future__ import annotations

import tempfile
import unittest
import base64
import io
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pymupdf as fitz
from PIL import Image

from app.services.ingestion import (
    IngestionError,
    _default_vision_provider,
    _normalize_openai_compatible_base_url,
    _render_page,
    chunk_documents,
    ingest_file,
)
from app.rag import vectorstore, workflow
from app.services import audit
from langchain_core.documents import Document


def test_settings(**overrides):
    values = {
        "ocr_provider": "none",
        "tesseract_cmd": "",
        "ocr_timeout_seconds": 5,
        "vision_provider": "openai_compatible",
        "vision_model": "",
        "vision_api_key": "",
        "vision_base_url": "",
        "vision_timeout_seconds": 5,
        "openrouter_api_key": "",
        "openrouter_base_url": "https://example.invalid/v1",
        "pdf_vision_max_pages": 8,
        "pdf_vision_min_image_area_ratio": 0.05,
        "pdf_vision_min_drawings": 80,
        "pdf_render_dpi": 100,
        "pdf_vision_max_pixels": 20_000_000,
        "pdf_vision_max_image_bytes": 12_582_912,
        "pdf_min_text_words": 25,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class MultimodalIngestionTests(unittest.TestCase):
    def test_vision_provider_reuses_and_overrides_openrouter_credentials(self):
        cases = [
            ({"vision_api_key": "", "openrouter_api_key": "router-key"}, "router-key"),
            ({"vision_api_key": "vision-key", "openrouter_api_key": "router-key"}, "vision-key"),
        ]
        for overrides, expected_key in cases:
            captured = {}

            class FakeChatOpenAI:
                def __init__(self, **kwargs):
                    captured.update(kwargs)

            values = {
                "vision_model": "nvidia/nemotron-3.5-super-vl:free",
                "openrouter_api_key": "router-key",
                "openrouter_base_url": "https://openrouter.ai/api/v1",
            }
            values.update(overrides)
            settings = test_settings(
                **values,
            )
            with patch("app.services.ingestion.ChatOpenAI", FakeChatOpenAI):
                provider = _default_vision_provider(settings)
            self.assertIsNotNone(provider)
            self.assertEqual(captured["api_key"], expected_key)
            self.assertEqual(captured["model"], "nvidia/nemotron-3.5-super-vl:free")

    def test_vision_provider_base_url_fallback_override_and_single_endpoint_path(self):
        self.assertEqual(
            _normalize_openai_compatible_base_url(
                "https://openrouter.ai/api/v1/chat/completions/"
            ),
            "https://openrouter.ai/api/v1",
        )
        self.assertEqual(
            _normalize_openai_compatible_base_url(
                "https://example.test/v1/chat/completions/chat/completions"
            ),
            "https://example.test/v1",
        )
        for vision_url, expected_url in [
            ("", "https://openrouter.ai/api/v1"),
            ("https://vision.example/v1/chat/completions", "https://vision.example/v1"),
        ]:
            captured = {}

            class FakeChatOpenAI:
                def __init__(self, **kwargs):
                    captured.update(kwargs)

            settings = test_settings(
                vision_model="vision-model",
                vision_api_key="vision-key",
                vision_base_url=vision_url,
                openrouter_base_url="https://openrouter.ai/api/v1",
            )
            with patch("app.services.ingestion.ChatOpenAI", FakeChatOpenAI):
                _default_vision_provider(settings)
            self.assertEqual(captured["base_url"], expected_url)
            self.assertNotIn("/chat/completions", captured["base_url"])

    def test_vision_client_sends_image_and_parses_successful_response(self):
        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured["config"] = kwargs

            def invoke(self, messages):
                captured["messages"] = messages
                return SimpleNamespace(content="A diagram shows VPN client → gateway.")

        settings = test_settings(
            vision_model="vision-model",
            vision_api_key="vision-key",
        )
        pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 2, 2), 0)
        pixmap.clear_with(200)
        png_bytes = pixmap.tobytes("png")
        with patch("app.services.ingestion.ChatOpenAI", FakeChatOpenAI):
            provider = _default_vision_provider(settings)
            result = provider(png_bytes, 2, "network.pdf")

        self.assertEqual(result, "A diagram shows VPN client → gateway.")
        message = captured["messages"][0]
        image_part = next(part for part in message.content if part["type"] == "image_url")
        image_url = image_part["image_url"]["url"]
        self.assertTrue(image_url.startswith("data:image/png;base64,"))
        decoded = base64.b64decode(image_url.split(",", 1)[1])
        with Image.open(io.BytesIO(decoded)) as sent_image:
            self.assertEqual(sent_image.format, "PNG")
        self.assertEqual(captured["config"]["max_retries"], 0)

    def test_page_rendering_respects_pixel_and_payload_limits(self):
        pdf = fitz.open()
        page = pdf.new_page(width=300, height=180)
        image_bytes = _render_page(page, 150, max_pixels=100, max_image_bytes=100_000)
        with Image.open(io.BytesIO(image_bytes)) as image:
            self.assertLessEqual(image.width * image.height, 100)
        with self.assertRaisesRegex(ValueError, "configured image limit"):
            _render_page(page, 72, max_pixels=100, max_image_bytes=1)
        pdf.close()

    def test_vision_timeout_http_and_malformed_responses_are_safe_failures(self):
        class RateLimitError(Exception):
            status_code = 429

        class FakeChatOpenAI:
            def __init__(self, *, failure=None, **kwargs):
                self.failure = failure

            def invoke(self, _messages):
                if self.failure:
                    raise self.failure
                return SimpleNamespace(content=[{"type": "image", "url": "private"}])

        settings = test_settings(vision_model="vision-model", vision_api_key="secret-key")
        failures = [TimeoutError("secret-key must not escape"), RateLimitError("rate limited")]
        for failure in failures:
            with patch(
                "app.services.ingestion.ChatOpenAI",
                lambda **kwargs: FakeChatOpenAI(failure=failure, **kwargs),
            ):
                provider = _default_vision_provider(settings)
                with self.assertRaises(Exception):
                    provider(b"image", 1, "doc.pdf")

        with patch("app.services.ingestion.ChatOpenAI", FakeChatOpenAI):
            provider = _default_vision_provider(settings)
            with self.assertRaisesRegex(ValueError, "malformed"):
                provider(b"image", 1, "doc.pdf")

    def test_empty_vision_response_reports_parse_stage_and_retains_ocr(self):
        class EmptyResponseModel:
            def __init__(self, **_kwargs):
                pass

            def invoke(self, _messages):
                return SimpleNamespace(content="")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "empty-vision-response.pdf"
            pdf = fitz.open()
            page = pdf.new_page(width=300, height=180)
            pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 300, 180), 0)
            pixmap.clear_with(245)
            page.insert_image(page.rect, stream=pixmap.tobytes("png"))
            pdf.save(path)
            pdf.close()

            settings = test_settings(
                vision_model="test/vision-model",
                vision_api_key="secret-key",
            )
            with (
                patch("app.services.ingestion.get_settings", return_value=settings),
                patch("app.services.ingestion.ChatOpenAI", EmptyResponseModel),
                self.assertLogs("app.services.ingestion", level="WARNING") as captured,
            ):
                result = ingest_file(
                    path,
                    source_filename="handbook.pdf",
                    ocr_provider=lambda _image: "OCR retained the visible troubleshooting text.",
                )

        self.assertEqual(result.vision_status, "failed")
        self.assertEqual(result.vision_failed_pages, [1])
        self.assertTrue(any("OCR retained" in doc.page_content for doc in result.documents))
        self.assertTrue(
            any("vision response parsing failed" in warning for warning in result.warnings)
        )
        record = captured.records[0]
        self.assertEqual(record.failed_stage, "response_parsing")
        self.assertEqual(record.exception_type, "ValueError")
        self.assertEqual(
            record.exception_message,
            "Vision provider returned an empty or malformed response",
        )
        self.assertEqual(record.provider_response["content_kind"], "text")
        self.assertNotIn("secret-key", repr(record.__dict__))

    def test_vision_description_is_chunked_indexed_and_retrievable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "diagram.pdf"
            pdf = fitz.open()
            page = pdf.new_page(width=300, height=180)
            page.draw_rect(fitz.Rect(15, 20, 285, 165))
            pdf.save(path)
            pdf.close()

            with patch(
                "app.services.ingestion.get_settings",
                return_value=test_settings(pdf_vision_min_drawings=1),
            ):
                result = ingest_file(
                    path,
                    source_filename="diagram.pdf",
                    vision_provider=lambda *_: "VPN client connects to gateway; arrow indicates authentication.",
                )
            description = next(
                document for document in result.documents
                if document.metadata["content_type"] == "diagram_description"
            )
            chunks = chunk_documents([description])
            fake_store = SimpleNamespace(
                add_documents=lambda docs, ids: ids,
                index=SimpleNamespace(delete=lambda **kwargs: None),
            )
            with patch.object(vectorstore, "get_vectorstore", return_value=fake_store):
                self.assertEqual(vectorstore.add_documents(chunks), [chunks[0].metadata["chunk_id"]])
            with patch.object(workflow, "get_retriever", return_value=SimpleNamespace(invoke=lambda _query: chunks)):
                retrieved = workflow.retrieve_kb({"current_query": "How does VPN auth work?"})["kb_docs"]

            self.assertIn("VPN client connects to gateway", retrieved[0].page_content)
            self.assertEqual(retrieved[0].metadata["source_filename"], "diagram.pdf")
            self.assertEqual(retrieved[0].metadata["page"], 1)
            self.assertTrue(retrieved[0].metadata["chunk_id"])

    def test_vision_provider_failures_are_reported_without_exception_text(self):
        class ProviderError(Exception):
            status_code = 429

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "diagram.pdf"
            pdf = fitz.open()
            page = pdf.new_page()
            for x in range(10, 200, 10):
                page.draw_rect(fitz.Rect(x, 50, x + 3, 100))
            pdf.save(path)
            pdf.close()

            def fail_vision(*_args):
                raise ProviderError("secret-key must not escape")

            with (
                patch(
                    "app.services.ingestion.get_settings",
                    return_value=test_settings(pdf_vision_min_drawings=1),
                ),
                self.assertLogs("app.services.ingestion", level="WARNING") as captured,
            ):
                result = ingest_file(path, vision_provider=fail_vision)

            self.assertEqual(result.vision_status, "failed")
            self.assertEqual(result.visual_pages_detected, [1])
            self.assertEqual(result.vision_pages, [])
            self.assertTrue(any("HTTP 429" in warning for warning in result.warnings))
            self.assertFalse(any("secret-key" in warning for warning in result.warnings))
            self.assertEqual(captured.records[0].failed_stage, "provider_request")
            self.assertEqual(captured.records[0].provider_status, 429)
            self.assertNotIn("secret-key", repr(captured.records[0].__dict__))

    def test_selectable_pdf_text_has_stable_provenance_and_chunk_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "guide.pdf"
            pdf = fitz.open()
            page = pdf.new_page()
            page.insert_text((72, 72), "VPN setup guide: open the secure client and sign in.")
            pdf.save(path)
            pdf.close()

            with patch("app.services.ingestion.get_settings", return_value=test_settings()):
                first = ingest_file(path, source_filename="guide.pdf")
                second = ingest_file(path, source_filename="guide.pdf")

            self.assertEqual(first.documents[0].metadata["page"], 1)
            self.assertEqual(first.documents[0].metadata["content_type"], "text")
            self.assertEqual(first.vision_status, "not_needed")
            self.assertEqual(first.documents[0].metadata["source_filename"], "guide.pdf")
            self.assertEqual(first.document_id, second.document_id)
            self.assertEqual(first.document_version, second.document_version)
            self.assertEqual(
                chunk_documents(first.documents)[0].metadata["chunk_id"],
                chunk_documents(second.documents)[0].metadata["chunk_id"],
            )

    def test_text_only_pdf_does_not_fail_or_warn_when_vision_credentials_are_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ordinary.pdf"
            pdf = fitz.open()
            pdf.new_page().insert_text((72, 72), "A normal text-only PDF guide.")
            pdf.save(path)
            pdf.close()
            settings = test_settings(
                vision_model="nvidia/nemotron-3.5-super-vl:free",
                vision_api_key="",
                openrouter_api_key="",
            )
            with patch("app.services.ingestion.get_settings", return_value=settings):
                result = ingest_file(path)

        self.assertTrue(result.documents)
        self.assertEqual(result.vision_status, "not_needed")
        self.assertFalse(any("VISION_API_KEY" in warning for warning in result.warnings))

    def test_scanned_page_uses_mock_ocr_and_vision_without_paid_access(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scan.pdf"
            pdf = fitz.open()
            page = pdf.new_page(width=300, height=180)
            pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 300, 180), 0)
            pixmap.clear_with(245)
            page.insert_image(page.rect, stream=pixmap.tobytes("png"))
            pdf.save(path)
            pdf.close()

            calls = {"ocr": 0, "vision": 0}

            def fake_ocr(_image_bytes):
                calls["ocr"] += 1
                return "Error 691: authentication failed. Check your VPN credentials."

            def fake_vision(_image_bytes, page_number, filename):
                calls["vision"] += 1
                self.assertEqual(page_number, 1)
                self.assertEqual(filename, "scan.pdf")
                return "A warning dialog shows error 691 and a Retry button."

            with patch("app.services.ingestion.get_settings", return_value=test_settings()):
                result = ingest_file(
                    path,
                    source_filename="scan.pdf",
                    ocr_provider=fake_ocr,
                    vision_provider=fake_vision,
                )

            content_types = {doc.metadata["content_type"] for doc in result.documents}
            self.assertIn("ocr", content_types)
            self.assertIn("screenshot_description", content_types)
            self.assertEqual(result.ocr_pages, [1])
            self.assertEqual(result.vision_pages, [1])
            self.assertEqual(calls, {"ocr": 1, "vision": 1})
            self.assertTrue(all(doc.metadata["page"] == 1 for doc in result.documents))

    def test_ocr_failure_is_reported_and_never_silently_discarded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "blank.pdf"
            pdf = fitz.open()
            pdf.new_page()
            pdf.save(path)
            pdf.close()

            def failing_ocr(_image_bytes):
                raise TimeoutError("synthetic timeout")

            with patch("app.services.ingestion.get_settings", return_value=test_settings()):
                result = ingest_file(path, ocr_provider=failing_ocr)

            self.assertEqual(result.documents, [])
            self.assertTrue(any("OCR failed" in warning for warning in result.warnings))
            self.assertTrue(any("No searchable text" in warning for warning in result.warnings))
            self.assertTrue(result.partial)

    def test_encrypted_and_malformed_pdfs_fail_with_clear_ingestion_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            encrypted_path = Path(temp_dir) / "encrypted.pdf"
            pdf = fitz.open()
            pdf.new_page()
            pdf.save(
                encrypted_path,
                encryption=fitz.PDF_ENCRYPT_AES_256,
                user_pw="reader-password",
                owner_pw="owner-password",
            )
            pdf.close()
            malformed_path = Path(temp_dir) / "malformed.pdf"
            malformed_path.write_bytes(b"not a pdf")

            with patch("app.services.ingestion.get_settings", return_value=test_settings()):
                with self.assertRaisesRegex(IngestionError, "encrypted"):
                    ingest_file(encrypted_path)
                with self.assertRaisesRegex(IngestionError, "open PDF"):
                    ingest_file(malformed_path)

    def test_text_fallback_and_changed_document_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "notes.txt"
            path.write_text("Initial troubleshooting notes", encoding="utf-8")
            with patch("app.services.ingestion.get_settings", return_value=test_settings()):
                first = ingest_file(path, source_filename="notes.txt")
                path.write_text("Updated troubleshooting notes", encoding="utf-8")
                second = ingest_file(path, source_filename="notes.txt")

            self.assertEqual(first.documents[0].metadata["content_type"], "text")
            self.assertEqual(first.document_id, second.document_id)
            self.assertNotEqual(first.document_version, second.document_version)
            self.assertNotEqual(
                chunk_documents(first.documents)[0].metadata["chunk_id"],
                chunk_documents(second.documents)[0].metadata["chunk_id"],
            )

    def test_vector_upsert_uses_chunk_ids_and_version_filters(self):
        class FakeIndex:
            def __init__(self):
                self.deletions = []

            def delete(self, **kwargs):
                self.deletions.append(kwargs)

        class FakeStore:
            def __init__(self):
                self.index = FakeIndex()
                self.add_call = None

            def add_documents(self, documents, ids):
                self.add_call = (documents, ids)
                return ids

        fake_store = FakeStore()
        document = Document(
            page_content="new page contents",
            metadata={"chunk_id": "stable-chunk-id"},
        )
        with patch.object(vectorstore, "get_vectorstore", return_value=fake_store):
            self.assertEqual(vectorstore.add_documents([document]), ["stable-chunk-id"])
            vectorstore.delete_obsolete_document_versions("doc-id", "new-version")
            vectorstore.delete_document_vectors("doc-id")

        self.assertEqual(fake_store.add_call[1], ["stable-chunk-id"])
        self.assertEqual(
            fake_store.index.deletions[0]["filter"],
            {"document_id": "doc-id", "document_version": {"$ne": "new-version"}},
        )
        self.assertEqual(
            fake_store.index.deletions[1]["filter"], {"document_id": "doc-id"}
        )

    def test_ingestion_audit_stores_safe_diagnostics_separately(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "audit.db"
            with patch.object(audit, "settings", SimpleNamespace(audit_db_path=str(db_path))):
                audit.init_db()
                audit.write_ingestion_audit(
                    "doc-id", "guide.pdf", "version", "partial", "partial", ["HTTP 429"]
                )
            connection = sqlite3.connect(db_path)
            try:
                row = connection.execute(
                    "SELECT document_id, status, vision_status, diagnostics_json "
                    "FROM ingestion_audit"
                ).fetchone()
            finally:
                connection.close()

        self.assertEqual(row[:3], ("doc-id", "partial", "partial"))
        self.assertEqual(row[3], '["HTTP 429"]')


if __name__ == "__main__":
    unittest.main()
