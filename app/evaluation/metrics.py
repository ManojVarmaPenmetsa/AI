"""Pure metric calculations, independent of providers and the application graph."""
from __future__ import annotations

import re
from typing import Any


def _source_page(item: dict[str, Any]) -> tuple[str, str | None]:
    return (str(item.get("source", "")).casefold(), str(item.get("page")) if item.get("page") is not None else None)


def retrieval_metrics(expected: list[dict[str, Any]], ranked: list[dict[str, Any]], k: int = 5) -> dict[str, float | None]:
    if not expected:
        return {f"recall_at_{k}": None, "mrr": None}
    targets = {_source_page(e) for e in expected}
    found = 0
    reciprocal_rank = 0.0
    for rank, item in enumerate(ranked[:k], start=1):
        if _source_page(item) in targets:
            found += 1
            if reciprocal_rank == 0:
                reciprocal_rank = 1 / rank
    return {f"recall_at_{k}": found / len(targets), "mrr": reciprocal_rank}


def citation_metrics(expected: list[dict[str, Any]], citations: list[dict[str, Any]]) -> dict[str, float | None]:
    if not citations:
        return {"citation_precision": 0.0 if expected else None, "citation_coverage": 0.0 if expected else None}
    expected_set = {_source_page(e) for e in expected}
    cited: set[tuple[str, str | None]] = set()
    for citation in citations:
        source = str(citation.get("source") or citation.get("title") or "").casefold()
        match = re.search(r"(.+?)\s*[—-]\s*page\s+(\d+)", source, flags=re.I)
        if match:
            cited.add((match.group(1).strip(), match.group(2)))
        else:
            cited.add((source, str(citation["page"]) if citation.get("page") is not None else None))
    true_hits = len(cited & expected_set)
    precision = true_hits / len(cited) if cited else 0.0
    coverage = true_hits / len(expected_set) if expected_set else None
    return {"citation_precision": precision, "citation_coverage": coverage}


def answer_fact_score(expected_facts: list[str], answer: str) -> float | None:
    if not expected_facts:
        return None
    normalize = lambda value: re.sub(r"\W+", " ", value.casefold()).strip()
    normalized_answer = normalize(answer)
    return sum(normalize(fact) in normalized_answer for fact in expected_facts) / len(expected_facts)


def aggregate_abstention(results: list[dict[str, Any]]) -> dict[str, float | None]:
    measurable = [r for r in results if r.get("should_abstain") is not None and r.get("abstained") is not None]
    if not measurable:
        return {"abstention_precision": None, "abstention_recall": None, "abstention_accuracy": None}
    tp = sum(r["should_abstain"] and r["abstained"] for r in measurable)
    fp = sum(not r["should_abstain"] and r["abstained"] for r in measurable)
    fn = sum(r["should_abstain"] and not r["abstained"] for r in measurable)
    correct = sum(r["should_abstain"] == r["abstained"] for r in measurable)
    return {
        "abstention_precision": tp / (tp + fp) if tp + fp else 0.0,
        "abstention_recall": tp / (tp + fn) if tp + fn else 0.0,
        "abstention_accuracy": correct / len(measurable),
    }
