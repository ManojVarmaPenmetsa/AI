"""Page-aware document ingestion with optional local OCR and configured vision."""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import pymupdf as fitz
from docx import Document as DocxDocument
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings


SUPPORTED = {".pdf", ".txt", ".md", ".docx"}
OCRProvider = Callable[[bytes], str]
VisionProvider = Callable[[bytes, int, str], str]
logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    documents: list[Document]
    document_id: str
    document_version: str
    source_filename: str
    ingestion_timestamp: str
    page_count: int = 1
    ocr_pages: list[int] = field(default_factory=list)
    vision_pages: list[int] = field(default_factory=list)
    vision_failed_pages: list[int] = field(default_factory=list)
    visual_pages_detected: list[int] = field(default_factory=list)
    vision_status: str = "not_needed"
    warnings: list[str] = field(default_factory=list)

    @property
    def partial(self) -> bool:
        return bool(self.warnings)


class IngestionError(ValueError):
    """Raised when the input cannot be read as its declared document type."""


class OCRConfigurationError(IngestionError):
    """Raised when OCR provider configuration is incomplete or invalid."""


class OCRSpaceAPIError(IngestionError):
    """Raised for safe, categorized OCR.space request and response failures."""

    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


class VisionAnalysisFailure(ValueError):
    """Vision failure tagged with the phase that produced it."""

    def __init__(self, stage: str, original: Exception, response_summary: dict | None = None):
        self.stage = stage
        self.original = original
        self.response_summary = response_summary
        super().__init__(str(original))


def stable_document_id(filename: str) -> str:
    normalized = unicodedata.normalize("NFKC", filename).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def content_version(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _base_metadata(
    source_filename: str, content: bytes, ingestion_timestamp: str,
    document_id: str | None = None, generation_id: str | None = None,
) -> dict[str, object]:
    metadata = {
        "source": source_filename,
        "source_filename": source_filename,
        "document_id": document_id or stable_document_id(source_filename),
        "document_version": content_version(content),
        "ingestion_timestamp": ingestion_timestamp,
    }
    if generation_id:
        metadata["generation_id"] = generation_id
    return metadata


def _document(
    content: str,
    metadata: dict[str, object],
) -> Document | None:
    cleaned = content.strip()
    return Document(page_content=cleaned, metadata=metadata) if cleaned else None


def _default_ocr_provider(settings) -> OCRProvider | None:
    provider = settings.ocr_provider.strip().lower()
    if provider in {"", "none", "off", "disabled"}:
        return None
    if provider != "ocr_space":
        raise IngestionError(f"Unsupported OCR_PROVIDER: {settings.ocr_provider}")

    api_key = settings.ocr_space_api_key.strip()
    if not api_key:
        raise OCRConfigurationError(
            "OCR_SPACE_API_KEY must be configured when OCR_PROVIDER=ocr_space."
        )
    api_url = settings.ocr_space_api_url.strip()
    if not api_url.startswith("https://"):
        raise OCRConfigurationError("OCR_SPACE_API_URL must use HTTPS.")
    max_file_bytes = int(settings.ocr_space_max_file_bytes)
    if max_file_bytes <= 0:
        raise OCRConfigurationError("OCR_SPACE_MAX_FILE_BYTES must be positive.")

    def ocr(image_bytes: bytes) -> str:
        import httpx

        prepared = _prepare_ocr_space_image(image_bytes, max_file_bytes)
        try:
            response = httpx.post(
                api_url,
                headers={"apikey": api_key},
                data={"OCREngine": str(settings.ocr_space_engine), "language": "eng"},
                files={"file": ("page.jpg", prepared, "image/jpeg")},
                timeout=settings.ocr_timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise OCRSpaceAPIError("OCR.space request timed out") from exc
        except httpx.RequestError as exc:
            raise OCRSpaceAPIError("OCR.space request failed") from exc

        status = response.status_code
        if status == 401 or status == 403:
            raise OCRSpaceAPIError("OCR.space rejected the API key", status)
        if status == 429:
            raise OCRSpaceAPIError("OCR.space rate limit exceeded", status)
        if status >= 500:
            raise OCRSpaceAPIError("OCR.space service is temporarily unavailable", status)
        if status >= 400:
            raise OCRSpaceAPIError("OCR.space rejected the request", status)
        try:
            payload = response.json()
        except (ValueError, TypeError) as exc:
            raise OCRSpaceAPIError("OCR.space returned malformed JSON", status) from exc
        if not isinstance(payload, dict):
            raise OCRSpaceAPIError("OCR.space returned a malformed response", status)
        error_value = payload.get("ErrorMessage", payload.get("ErrorDetails", ""))
        if isinstance(error_value, list):
            error_text = " ".join(str(item) for item in error_value).casefold()
        else:
            error_text = str(error_value).casefold()
        if any(token in error_text for token in ("invalid api key", "incorrect api key", "api key is not valid", "apikey is invalid")):
            raise OCRSpaceAPIError("OCR.space rejected the API key", status)
        if any(token in error_text for token in ("rate limit", "request limit", "quota exceeded", "too many requests")):
            raise OCRSpaceAPIError("OCR.space rate limit exceeded", status)
        exit_code = payload.get("OCRExitCode")
        if payload.get("IsErroredOnProcessing") is True or (
            isinstance(exit_code, int) and exit_code not in {1, 2}
        ):
            suffix = f" (exit code {exit_code})" if isinstance(exit_code, int) else ""
            raise OCRSpaceAPIError(f"OCR.space could not process the image{suffix}", status)
        results = payload.get("ParsedResults")
        if not isinstance(results, list):
            raise OCRSpaceAPIError("OCR.space response is missing parsed results", status)
        extracted: list[str] = []
        for result in results:
            if not isinstance(result, dict) or not isinstance(result.get("ParsedText", ""), str):
                raise OCRSpaceAPIError("OCR.space returned malformed parsed text", status)
            text = result.get("ParsedText", "").strip()
            if text:
                extracted.append(text)
        return "\n\n".join(extracted)

    return ocr


def _prepare_ocr_space_image(image_bytes: bytes, max_file_bytes: int) -> bytes:
    """Encode one image as a size-bounded JPEG for the free API tier."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("RGB")
    except Exception as exc:
        raise OCRSpaceAPIError("OCR input is not a supported image") from exc

    for _ in range(12):
        for quality in (88, 78, 68, 58):
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=True)
            prepared = output.getvalue()
            if len(prepared) <= max_file_bytes:
                return prepared
        width = max(1, int(image.width * 0.85))
        height = max(1, int(image.height * 0.85))
        if (width, height) == image.size:
            break
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    raise OCRSpaceAPIError(
        f"OCR input exceeds the configured {max_file_bytes}-byte OCR.space file limit"
    )


def _default_vision_provider(settings) -> VisionProvider | None:
    model_name = settings.vision_model.strip()
    if not model_name:
        return None
    api_key = settings.vision_api_key.strip() or settings.openrouter_api_key.strip()
    if not api_key:
        return None
    if settings.vision_provider.strip().lower() != "openai_compatible":
        raise IngestionError(
            f"Unsupported VISION_PROVIDER: {settings.vision_provider}"
        )

    model = ChatOpenAI(
        model=model_name,
        base_url=_normalize_openai_compatible_base_url(
            settings.vision_base_url.strip() or settings.openrouter_base_url.strip()
        ),
        api_key=api_key,
        temperature=0,
        timeout=settings.vision_timeout_seconds,
        max_retries=0,
    )

    def describe(image_bytes: bytes, page_number: int, source_filename: str) -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Describe only information visibly present on this document page. "
                        "Focus on layout, visual relationships, and what arrows or chart "
                        "structure show. Avoid repeating text that OCR can extract; mention "
                        "a short label only when needed to explain a relationship. Do not "
                        "infer missing details. Preserve critical visual warnings or states. "
                        f"Source: {source_filename}, page {page_number}."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded}",
                        "detail": "high",
                    },
                },
            ]
        )
        try:
            response = model.invoke([message])
        except Exception as exc:
            raise VisionAnalysisFailure("provider_request", exc) from exc

        content = getattr(response, "content", None)
        response_summary = _response_shape(content)
        try:
            if isinstance(content, str):
                description = content.strip()
            elif isinstance(content, list):
                description = "\n".join(
                    str(part.get("text", "")).strip()
                    for part in content
                    if isinstance(part, dict)
                    and part.get("type") in {"text", "output_text"}
                    and part.get("text")
                ).strip()
            else:
                description = ""
            if not description:
                raise ValueError("Vision provider returned an empty or malformed response")
            return description
        except Exception as exc:
            raise VisionAnalysisFailure(
                "response_parsing", exc, response_summary=response_summary
            ) from exc

    return describe


def _normalize_openai_compatible_base_url(base_url: str) -> str:
    """Return the API root; OpenAI SDK appends /chat/completions for us."""
    normalized = base_url.strip().rstrip("/")
    suffix = "/chat/completions"
    while normalized.lower().endswith(suffix):
        normalized = normalized[: -len(suffix)].rstrip("/")
    return normalized


def _safe_provider_error(exc: Exception) -> str:
    """Describe provider failures without including request or response contents."""
    if isinstance(exc, OCRConfigurationError):
        return str(exc)
    if isinstance(exc, OCRSpaceAPIError):
        if exc.status_code is not None:
            return f"OCRSpaceAPIError (HTTP {exc.status_code}): {exc}"
        return f"OCRSpaceAPIError: {exc}"
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        status = getattr(exc, "status_code", None)
    if status is not None:
        return f"{type(exc).__name__} (HTTP {status})"
    return type(exc).__name__


def _response_shape(content: object) -> dict[str, object]:
    """Describe response structure only; never include returned text or image data."""
    if isinstance(content, str):
        return {"content_kind": "text", "content_length": len(content)}
    if isinstance(content, list):
        return {
            "content_kind": "parts",
            "part_count": len(content),
            "part_types": [
                part_type
                if isinstance(part_type, str)
                and part_type in {"text", "output_text", "image", "refusal"}
                else "other"
                for part in content[:12]
                if isinstance(part, dict)
                for part_type in [part.get("type")]
            ],
        }
    return {"content_kind": type(content).__name__ if content is not None else "missing"}


def _sanitize_diagnostic_message(message: object, sensitive_values: tuple[str, ...] = ()) -> str:
    """Bound diagnostics and remove common credential, data URL, and base64 forms."""
    text = str(message or "").replace("\r", " ").replace("\n", " ")
    for value in sensitive_values:
        if value:
            text = text.replace(value, "[REDACTED]")
    patterns = (
        (r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]"),
        (r"(?i)\b(?:sk-or-v1-|sk-|or-v1-)[A-Za-z0-9_-]{8,}", "[REDACTED_KEY]"),
        (r"(?i)(?:api[_ -]?key|authorization|password|access[_ -]?token)\s*[:=]\s*[^\s,;]+", "[REDACTED_CREDENTIAL]"),
        (r"data:image/[^;\s]+;base64,[A-Za-z0-9+/=]+", "data:image/[REDACTED]"),
        (r"(?<![A-Za-z0-9])[A-Za-z0-9+/]{120,}={0,2}(?![A-Za-z0-9])", "[REDACTED_DATA]"),
        (r"(?i)secret[-_ ]?key", "[REDACTED]"),
    )
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    return text[:300] + ("..." if len(text) > 300 else "")


def _provider_error_details(
    exc: Exception, sensitive_values: tuple[str, ...] = ()
) -> tuple[int | None, dict[str, str]]:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(exc, "status_code", None)
    details: dict[str, str] = {}
    # Only allowlisted error fields are retained. Headers and request/response
    # bodies are deliberately never logged.
    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        for field_name in ("code", "type", "message"):
            value = error.get(field_name)
            if value is not None:
                details[field_name] = _sanitize_diagnostic_message(value, sensitive_values)
    return (int(status) if isinstance(status, int) else None), details


def _log_vision_failure(
    *, document_id: str, page_number: int, stage: str, exc: Exception,
    response_summary: dict | None = None, sensitive_values: tuple[str, ...] = (),
) -> tuple[int | None, str]:
    original = exc.original if isinstance(exc, VisionAnalysisFailure) else exc
    if isinstance(exc, VisionAnalysisFailure):
        stage = exc.stage
        response_summary = exc.response_summary or response_summary
    status, provider_response = _provider_error_details(original, sensitive_values)
    message = _sanitize_diagnostic_message(str(original), sensitive_values)
    logger.warning(
        "PDF vision analysis failed",
        extra={
            "event": "pdf_vision_analysis_failed",
            "document_id": document_id,
            "page": page_number,
            "failed_stage": stage,
            "exception_type": type(original).__name__,
            "exception_message": message,
            "provider_status": status,
            "provider_response": provider_response or response_summary,
        },
    )
    return status, message


def _render_page(
    page: fitz.Page,
    dpi: int,
    *,
    max_pixels: int = 20_000_000,
    max_image_bytes: int = 12_582_912,
) -> bytes:
    page_area = max(1.0, page.rect.width * page.rect.height)
    scale = min(max(0.1, dpi / 72.0), math.sqrt(max_pixels / page_area))
    if scale <= 0:
        raise ValueError("Page dimensions exceed the safe rendering limit")
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    for _ in range(10):
        if pixmap.width * pixmap.height <= max_pixels:
            break
        scale *= 0.9
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    if pixmap.width * pixmap.height > max_pixels:
        raise ValueError("Rendered page exceeds the configured pixel limit")
    image_bytes = pixmap.tobytes("png")
    if len(image_bytes) > max_image_bytes:
        raise ValueError(
            f"Rendered page exceeds the configured image limit ({max_image_bytes} bytes)"
        )
    return image_bytes


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def _page_table_text(page: fitz.Page) -> list[str]:
    find_tables = getattr(page, "find_tables", None)
    if not callable(find_tables):
        return []
    found = find_tables()
    output: list[str] = []
    for table in getattr(found, "tables", []):
        rows = table.extract() or []
        lines = [" | ".join(str(cell or "").strip() for cell in row) for row in rows]
        rendered = "\n".join(line for line in lines if line.replace("|", "").strip())
        if rendered:
            output.append(rendered)
    return output


def _pdf_documents(
    path: Path,
    base: dict[str, object],
    settings,
    warnings: list[str],
    ocr_pages: list[int],
    vision_pages: list[int],
    vision_failed_pages: list[int],
    ocr_provider: OCRProvider | None,
    vision_provider: VisionProvider | None,
) -> tuple[list[Document], int]:
    try:
        pdf = fitz.open(path)
    except Exception as exc:
        raise IngestionError(f"Could not open PDF: {type(exc).__name__}: {exc}") from exc

    with pdf:
        if pdf.needs_pass and not pdf.authenticate(""):
            raise IngestionError("This PDF is encrypted and requires a password.")
        if pdf.page_count == 0:
            raise IngestionError("This PDF has no pages.")

        selectable_text: dict[int, str] = {}
        try:
            for item in PyPDFLoader(str(path)).load():
                page_index = item.metadata.get("page")
                if isinstance(page_index, int):
                    selectable_text[page_index + 1] = item.page_content
        except Exception as exc:
            warnings.append(
                "The standard PDF text loader failed; PyMuPDF page text was used "
                f"where available ({type(exc).__name__})."
            )

        docs: list[Document] = []
        visual_candidates: list[tuple[int, bytes, str]] = []
        visual_pages_detected: list[int] = []
        for index in range(pdf.page_count):
            page_number = index + 1
            page = pdf.load_page(index)
            page_meta = {**base, "page": page_number, "page_label": str(page_number)}
            text = selectable_text.get(page_number, "").strip()
            if not text:
                try:
                    text = page.get_text("text", sort=True).strip()
                except Exception as exc:
                    warnings.append(
                        f"Page {page_number}: selectable text extraction failed "
                        f"({type(exc).__name__})."
                    )

            text_doc = _document(
                text,
                {**page_meta, "content_type": "text", "provenance": "selectable_pdf_text"},
            )
            if text_doc:
                docs.append(text_doc)

            try:
                for table_no, table_text in enumerate(_page_table_text(page), 1):
                    table_doc = _document(
                        table_text,
                        {
                            **page_meta,
                            "content_type": "table",
                            "table_number": table_no,
                            "provenance": "pymupdf_table_detection",
                        },
                    )
                    if table_doc:
                        docs.append(table_doc)
            except Exception as exc:
                warnings.append(
                    f"Page {page_number}: table extraction failed "
                    f"({type(exc).__name__}); ordinary page text was retained."
                )

            try:
                image_info = page.get_image_info(xrefs=True)
            except Exception as exc:
                image_info = []
                warnings.append(
                    f"Page {page_number}: embedded image inspection failed "
                    f"({type(exc).__name__})."
                )
            page_area = max(1.0, page.rect.width * page.rect.height)
            image_area_ratio = min(
                1.0,
                sum(
                    max(0.0, info["bbox"][2] - info["bbox"][0])
                    * max(0.0, info["bbox"][3] - info["bbox"][1])
                    for info in image_info
                )
                / page_area,
            )
            try:
                drawings = page.get_drawings()
            except Exception as exc:
                drawings = []
                warnings.append(
                    f"Page {page_number}: vector graphics inspection failed "
                    f"({type(exc).__name__})."
                )
            words = _word_count(text)
            sparse_page = words < settings.pdf_min_text_words

            if sparse_page and ocr_provider:
                try:
                    ocr_text = ocr_provider(
                        _render_page(
                            page,
                            settings.pdf_render_dpi,
                            max_pixels=settings.pdf_vision_max_pixels,
                            max_image_bytes=settings.pdf_vision_max_image_bytes,
                        )
                    ).strip()
                    if ocr_text:
                        docs.append(
                            Document(
                                page_content=ocr_text,
                                metadata={
                                    **page_meta,
                                    "content_type": "ocr",
                                    "provenance": "ocr_space_full_page_ocr",
                                },
                            )
                        )
                        ocr_pages.append(page_number)
                    else:
                        warnings.append(f"Page {page_number}: OCR returned no readable text.")
                except Exception as exc:
                    warnings.append(
                        f"Page {page_number}: OCR failed ({_safe_provider_error(exc)})."
                    )
            elif sparse_page and not ocr_provider:
                warnings.append(
                    f"Page {page_number}: little selectable text was found and OCR is disabled."
                )

            # OCR image crops on image-bearing pages with selectable body text;
            # sparse pages are OCRed once as a complete page above.
            if not sparse_page and image_info and ocr_provider:
                seen_xrefs: set[int] = set()
                for info in image_info:
                    xref = int(info.get("xref") or 0)
                    if not xref or xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)
                    try:
                        image = pdf.extract_image(xref)
                        image_bytes = image["image"]
                        if len(image_bytes) > settings.pdf_vision_max_image_bytes:
                            warnings.append(
                                f"Page {page_number}, image {xref}: extracted image exceeds "
                                "the configured processing limit; OCR was skipped."
                            )
                            continue
                        image_text = ocr_provider(image_bytes).strip()
                        if image_text:
                            docs.append(
                                Document(
                                    page_content=image_text,
                                    metadata={
                                        **page_meta,
                                        "content_type": "screenshot_ocr",
                                        "image_xref": xref,
                                        "provenance": "embedded_image_ocr",
                                    },
                                )
                            )
                            ocr_pages.append(page_number)
                        else:
                            warnings.append(
                                f"Page {page_number}, image {xref}: OCR returned no readable text."
                            )
                    except Exception as exc:
                        warnings.append(
                            f"Page {page_number}, image {xref}: OCR/extraction failed "
                            f"({_safe_provider_error(exc)})."
                        )
            elif not sparse_page and image_info and not ocr_provider:
                warnings.append(
                    f"Page {page_number}: embedded image(s) were detected, but OCR is disabled."
                )

            is_visual = bool(
                image_area_ratio >= settings.pdf_vision_min_image_area_ratio
                or len(drawings) >= settings.pdf_vision_min_drawings
            )
            if is_visual:
                visual_pages_detected.append(page_number)
                visual_type = (
                    "screenshot_description"
                    if image_area_ratio >= settings.pdf_vision_min_image_area_ratio
                    else "diagram_description"
                )
                if vision_provider:
                    try:
                        visual_candidates.append(
                            (
                                page_number,
                                _render_page(
                                    page,
                                    settings.pdf_render_dpi,
                                    max_pixels=settings.pdf_vision_max_pixels,
                                    max_image_bytes=settings.pdf_vision_max_image_bytes,
                                ),
                                visual_type,
                            )
                        )
                    except Exception as exc:
                        _log_vision_failure(
                            document_id=str(base["document_id"]),
                            page_number=page_number,
                            stage="image_preprocessing",
                            exc=exc,
                            sensitive_values=(
                                settings.vision_api_key.strip(),
                                settings.openrouter_api_key.strip(),
                                str(base["source_filename"]),
                            ),
                        )
                        vision_failed_pages.append(page_number)
                        safe_message = _sanitize_diagnostic_message(
                            str(exc),
                            (settings.vision_api_key.strip(), settings.openrouter_api_key.strip()),
                        )
                        warnings.append(
                            f"Page {page_number}: image preprocessing failed ({safe_message}); "
                            "text/OCR content was retained."
                        )
                else:
                    if settings.vision_model.strip() and not (
                        settings.vision_api_key.strip()
                        or settings.openrouter_api_key.strip()
                    ):
                        reason = (
                            "a vision model is configured, but VISION_API_KEY and "
                            "OPENROUTER_API_KEY are missing"
                        )
                    else:
                        reason = "no vision model is configured"
                    warnings.append(
                        f"Page {page_number}: visual content was detected, but {reason}; "
                        "extracted text/OCR was retained."
                    )

        max_vision_pages = max(0, settings.pdf_vision_max_pages)
        if len(visual_candidates) > max_vision_pages:
            skipped = [
                number for number, _, _ in visual_candidates[max_vision_pages:]
            ]
            warnings.append(
                "Vision page limit reached; visual analysis skipped for page(s): "
                + ", ".join(map(str, skipped))
            )
        for page_number, rendered_page, visual_type in visual_candidates[:max_vision_pages]:
            try:
                raw_description = vision_provider(
                    rendered_page, page_number, str(base["source_filename"])
                )
                if not isinstance(raw_description, str):
                    raise VisionAnalysisFailure(
                        "response_parsing",
                        ValueError("Vision provider returned a non-text description"),
                        response_summary=_response_shape(raw_description),
                    )
                description = raw_description.strip()
                if description:
                    docs.append(
                        Document(
                            page_content=(
                                f"Visual description of page {page_number} from "
                                f"{base['source_filename']}:\n{description}"
                            ),
                            metadata={
                                **base,
                                "page": page_number,
                                "page_label": str(page_number),
                                "content_type": visual_type,
                                "provenance": "vision_model_full_page_render",
                                "vision_model": settings.vision_model,
                            },
                        )
                    )
                    vision_pages.append(page_number)
                else:
                    error = ValueError("Vision provider returned an empty or malformed response")
                    _log_vision_failure(
                        document_id=str(base["document_id"]), page_number=page_number,
                        stage="response_parsing", exc=error,
                        response_summary={"content_kind": "empty_description"},
                    )
                    vision_failed_pages.append(page_number)
                    warnings.append(
                        f"Page {page_number}: vision response parsing failed "
                        "(provider returned an empty or malformed response); "
                        "text/OCR content was retained."
                    )
            except Exception as exc:
                status, _ = _log_vision_failure(
                    document_id=str(base["document_id"]),
                    page_number=page_number,
                    stage="provider_request",
                    exc=exc,
                    sensitive_values=(
                        settings.vision_api_key.strip(),
                        settings.openrouter_api_key.strip(),
                        str(base["source_filename"]),
                    ),
                )
                vision_failed_pages.append(page_number)
                if isinstance(exc, VisionAnalysisFailure) and exc.stage == "response_parsing":
                    category = "vision response parsing failed"
                    detail = "provider returned an empty or malformed response"
                else:
                    category = "vision provider request failed"
                    detail = type(exc.original if isinstance(exc, VisionAnalysisFailure) else exc).__name__
                    if status is not None:
                        detail += f", HTTP {status}"
                warnings.append(
                    f"Page {page_number}: {category} ({detail}); text/OCR content was retained."
                )

        return docs, pdf.page_count, visual_pages_detected


def _non_pdf_documents(
    path: Path,
    base: dict[str, object],
    warnings: list[str],
) -> list[Document]:
    suffix = path.suffix.lower()
    try:
        if suffix in {".txt", ".md"}:
            loaded = TextLoader(str(path), encoding="utf-8").load()
            content = "\n\n".join(item.page_content for item in loaded)
            doc = _document(
                content,
                {
                    **base,
                    "page": 1,
                    "page_label": "1",
                    "content_type": "text",
                    "provenance": "text_file",
                },
            )
            return [doc] if doc else []

        if suffix == ".docx":
            word_doc = DocxDocument(str(path))
            output: list[Document] = []
            text = "\n".join(p.text for p in word_doc.paragraphs if p.text.strip())
            doc = _document(
                text,
                {
                    **base,
                    "page": 1,
                    "page_label": "1",
                    "content_type": "text",
                    "provenance": "docx_paragraphs",
                },
            )
            if doc:
                output.append(doc)
            for table_no, table in enumerate(word_doc.tables, 1):
                table_text = "\n".join(
                    " | ".join(cell.text.strip() for cell in row.cells)
                    for row in table.rows
                )
                table_doc = _document(
                    table_text,
                    {
                        **base,
                        "page": 1,
                        "page_label": "1",
                        "content_type": "table",
                        "table_number": table_no,
                        "provenance": "docx_table",
                    },
                )
                if table_doc:
                    output.append(table_doc)
            return output
    except Exception as exc:
        raise IngestionError(
            f"Could not read {suffix} document: {type(exc).__name__}: {exc}"
        ) from exc
    raise IngestionError(f"Unsupported file type: {suffix}")


def ingest_file(
    path: Path,
    *,
    source_filename: str | None = None,
    document_id: str | None = None,
    generation_id: str | None = None,
    ocr_provider: OCRProvider | None = None,
    vision_provider: VisionProvider | None = None,
) -> IngestionResult:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise IngestionError(f"Unsupported file type: {suffix}")
    if not path.is_file():
        raise IngestionError("The uploaded document could not be found on disk.")
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise IngestionError(f"Could not read the uploaded document: {exc}") from exc
    if not content:
        raise IngestionError("The uploaded document is empty.")

    source = source_filename or path.name
    timestamp = datetime.now(timezone.utc).isoformat()
    base = _base_metadata(
        source, content, timestamp, document_id=document_id, generation_id=generation_id
    )
    settings = get_settings()
    warnings: list[str] = []
    ocr_pages: list[int] = []
    vision_pages: list[int] = []
    vision_failed_pages: list[int] = []
    visual_pages_detected: list[int] = []

    try:
        active_ocr = (
            ocr_provider
            if ocr_provider is not None
            else _default_ocr_provider(settings)
        )
    except Exception as exc:
        warnings.append(f"OCR provider setup failed ({_safe_provider_error(exc)}).")
        active_ocr = None
    try:
        active_vision = (
            vision_provider
            if vision_provider is not None
            else _default_vision_provider(settings)
        )
    except Exception as exc:
        warnings.append(f"Vision provider setup failed ({_safe_provider_error(exc)}).")
        active_vision = None

    if suffix == ".pdf":
        documents, page_count, visual_pages_detected = _pdf_documents(
            path,
            base,
            settings,
            warnings,
            ocr_pages,
            vision_pages,
            vision_failed_pages,
            active_ocr,
            active_vision,
        )
    else:
        documents = _non_pdf_documents(path, base, warnings)
        page_count = 1

    if not visual_pages_detected:
        vision_status = "not_needed"
    elif active_vision is None:
        vision_status = "not_configured"
    elif len(vision_pages) == len(visual_pages_detected):
        vision_status = "completed"
    elif vision_failed_pages and len(set(vision_failed_pages)) >= len(visual_pages_detected):
        vision_status = "failed"
    else:
        vision_status = "partial"

    if not documents:
        warnings.append(
            "No searchable text, OCR text, or visual description was produced; "
            "the previous indexed version, if any, was left unchanged."
        )
    return IngestionResult(
        documents=documents,
        document_id=str(base["document_id"]),
        document_version=str(base["document_version"]),
        source_filename=source,
        ingestion_timestamp=timestamp,
        page_count=page_count,
        ocr_pages=sorted(set(ocr_pages)),
        vision_pages=sorted(set(vision_pages)),
        vision_failed_pages=sorted(set(vision_failed_pages)),
        visual_pages_detected=sorted(set(visual_pages_detected)),
        vision_status=vision_status,
        warnings=warnings,
    )


def load_file(path: Path) -> list[Document]:
    """Backward-compatible text and document loader used by existing callers."""
    return ingest_file(path).documents


def chunk_documents(docs: Iterable[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        add_start_index=True,
    )
    chunks = splitter.split_documents(list(docs))
    for chunk in chunks:
        metadata = chunk.metadata
        identity = "|".join(
            str(metadata.get(key, ""))
            for key in (
                "document_id",
                "document_version",
                "generation_id",
                "page",
                "content_type",
                "start_index",
                "image_xref",
                "table_number",
                "provenance",
            )
        )
        chunk.metadata["chunk_id"] = hashlib.sha256(
            f"{identity}|{content_version(chunk.page_content.encode('utf-8'))}".encode("utf-8")
        ).hexdigest()
    return chunks
