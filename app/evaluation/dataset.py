"""Load and validate the small, versioned RAG evaluation dataset."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_CASE_FIELDS = {
    "id", "question", "expected_facts", "expected_evidence", "should_abstain",
    "expected_source_used", "offline",
}


def load_dataset(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_dataset(data)
    return data


def validate_dataset(data: Any) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("dataset_version"), str):
        raise ValueError("Dataset must be an object with a string dataset_version")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset cases must be a non-empty list")
    ids: set[str] = set()
    for index, case in enumerate(cases):
        prefix = f"cases[{index}]"
        if not isinstance(case, dict) or not REQUIRED_CASE_FIELDS.issubset(case):
            raise ValueError(f"{prefix} is missing required fields")
        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id.strip() or case_id in ids:
            raise ValueError(f"{prefix}.id must be a unique non-empty string")
        ids.add(case_id)
        if not isinstance(case["question"], str) or not case["question"].strip():
            raise ValueError(f"{prefix}.question must be non-empty")
        if not isinstance(case["expected_facts"], list) or not all(
            isinstance(item, str) and item.strip() for item in case["expected_facts"]
        ):
            raise ValueError(f"{prefix}.expected_facts must contain non-empty strings")
        if not isinstance(case["expected_evidence"], list):
            raise ValueError(f"{prefix}.expected_evidence must be a list")
        for evidence in case["expected_evidence"]:
            if not isinstance(evidence, dict) or not isinstance(evidence.get("source"), str):
                raise ValueError(f"{prefix}.expected_evidence entries need a source")
            if evidence.get("page") is not None and not isinstance(evidence["page"], (int, float)):
                raise ValueError(f"{prefix}.expected_evidence.page must be numeric or null")
        if not isinstance(case["should_abstain"], bool):
            raise ValueError(f"{prefix}.should_abstain must be boolean")
        if case["expected_source_used"] not in {"private_kb", "web_search", "insufficient_evidence"}:
            raise ValueError(f"{prefix}.expected_source_used is not supported")
        if not isinstance(case["offline"], dict) or not isinstance(case["offline"].get("retrieved_chunks"), list):
            raise ValueError(f"{prefix}.offline.retrieved_chunks must be a list")
        for chunk in case["offline"]["retrieved_chunks"]:
            if not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str) or not isinstance(chunk.get("metadata"), dict):
                raise ValueError(f"{prefix} has an invalid offline retrieved chunk")
        if case.get("history") is not None and not isinstance(case["history"], list):
            raise ValueError(f"{prefix}.history must be a list")
