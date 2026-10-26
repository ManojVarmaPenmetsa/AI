"""Safe classification and bounded retry helpers for OpenRouter responses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import logging
import re
from time import sleep


logger = logging.getLogger(__name__)
MAX_TRANSIENT_RETRIES = 1
MAX_RETRY_DELAY_SECONDS = 8.0


@dataclass(frozen=True)
class OpenRouterFailure:
    status_code: int | None
    category: str
    provider: str | None = None
    retry_after: str | None = None


def _response_body(exc: Exception) -> dict:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return body
    response = getattr(exc, "response", None)
    try:
        body = response.json() if response is not None else None
    except Exception:
        body = None
    if isinstance(body, dict):
        return body
    if isinstance(body, str):
        try:
            parsed = json.loads(body)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError):
            pass
    return {}


def _error_parts(body: dict) -> tuple[dict, dict]:
    error = body.get("error")
    error = error if isinstance(error, dict) else {}
    metadata = error.get("metadata")
    if not isinstance(metadata, dict):
        metadata = body.get("metadata")
    return error, metadata if isinstance(metadata, dict) else {}


def _safe_label(value: object) -> str | None:
    if not isinstance(value, (str, int, float)):
        return None
    label = re.sub(r"[^A-Za-z0-9_.:-]", "", str(value))[:80]
    return label or None


def classify_openrouter_failure(exc: Exception) -> OpenRouterFailure:
    """Classify by status and allowlisted metadata; never expose raw response text."""
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None

    retry_after = http_retry_after(exc)

    body = _response_body(exc)
    error, metadata = _error_parts(body)
    message = str(error.get("message") or "").casefold()
    code = str(error.get("code") or "").casefold()
    account_limit_markers = (
        "free-models-per-min", "free-models-per-day", "daily limit", "daily quota",
        "spending limit", "account limit", "insufficient credits", "quota exceeded",
        "credit limit", "credit balance", "rate limit exceeded: free",
    )
    provider = _safe_label(metadata.get("provider_name") or metadata.get("provider_slug"))
    if status == 429:
        if any(marker in message or marker in code for marker in account_limit_markers):
            category = "openrouter_account_or_model_limit"
        elif provider:
            category = "upstream_provider_rate_limit"
        else:
            category = "rate_limit_source_unknown"
    elif (
        status in {408, 409}
        or (status is not None and 500 <= status <= 599)
        or any(token in type(exc).__name__.casefold() for token in ("timeout", "connection"))
    ):
        category = "transient_upstream_error"
        provider = provider or _safe_label(metadata.get("provider"))
    else:
        category = "openrouter_error"
        provider = provider or _safe_label(metadata.get("provider"))

    return OpenRouterFailure(status, category, provider, retry_after)


def _retry_delay(failure: OpenRouterFailure, retry_number: int) -> float | None:
    """Return a bounded delay for transient errors; None means do not retry."""
    retryable = failure.category == "transient_upstream_error"
    retryable = retryable or failure.category == "upstream_provider_rate_limit"
    if not retryable:
        return None
    if failure.retry_after is not None:
        try:
            delay = float(failure.retry_after)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(failure.retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                return None
    else:
        delay = min(0.5 * (2 ** retry_number), MAX_RETRY_DELAY_SECONDS)
    if delay < 0 or delay > MAX_RETRY_DELAY_SECONDS:
        return None
    return delay


def invoke_with_transient_retries(runnable, prompt):
    """Call a LangChain runnable with one bounded retry for transient failures."""
    for retry_number in range(MAX_TRANSIENT_RETRIES + 1):
        try:
            return runnable.invoke(prompt)
        except Exception as exc:
            failure = classify_openrouter_failure(exc)
            delay = _retry_delay(failure, retry_number)
            if delay is None or retry_number >= MAX_TRANSIENT_RETRIES:
                if failure.status_code == 429:
                    logger.warning(
                        "OpenRouter rate limited category=%s provider=%s retry_after=%s",
                        failure.category,
                        failure.provider or "unknown",
                        failure.retry_after or "unspecified",
                    )
                raise
            logger.warning(
                "Transient OpenRouter failure category=%s provider=%s; retrying once after %.1fs",
                failure.category,
                failure.provider or "unknown",
                delay,
            )
            sleep(delay)


def http_retry_after(exc: Exception) -> str | None:
    """Return a safe Retry-After response value, including valid HTTP dates."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
    except AttributeError:
        return None
    if value is None:
        return None
    raw = str(value).strip()
    if re.fullmatch(r"\d{1,6}", raw):
        return raw
    try:
        parsed = parsedate_to_datetime(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        if parsed > datetime.now(timezone.utc):
            return parsed.strftime("%a, %d %b %Y %H:%M:%S GMT")
    except (TypeError, ValueError, OverflowError):
        pass
    return None
