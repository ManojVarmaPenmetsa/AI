"""Run the existing LangGraph workflow in offline fixture or explicit live mode."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from app.core.config import get_settings
from app.evaluation.metrics import (
    aggregate_abstention, answer_fact_score, citation_metrics, retrieval_metrics,
)
from app.rag import workflow


def _prompt_text(prompt: Any) -> str:
    if isinstance(prompt, list):
        return "\n".join(str(getattr(message, "content", message)) for message in prompt)
    return str(prompt)


class _FixtureRetriever:
    def __init__(self, case: dict[str, Any], record: list[dict[str, Any]]):
        self.case, self.record = case, record

    def invoke(self, query: str) -> list[Document]:
        chunks = self.case["offline"]["retrieved_chunks"]
        docs = [Document(page_content=c["text"], metadata=dict(c["metadata"])) for c in chunks]
        self.record[:] = [
            {
                "source": str(d.metadata.get("source_filename") or d.metadata.get("source") or "unknown"),
                "page": d.metadata.get("page"),
                "content_type": d.metadata.get("content_type"),
                "chunk_id": d.metadata.get("chunk_id"),
            }
            for d in docs
        ]
        return docs


class _FixtureLLM:
    def __init__(self, case: dict[str, Any]):
        self.case = case

    def with_structured_output(self, schema, **kwargs):
        outer = self

        class Structured:
            def invoke(self, prompt: str):
                prompt = _prompt_text(prompt)
                if schema is workflow.RouteDecision:
                    return schema(route="kb")
                if '"web_results"' in prompt or '"web_evidence"' in prompt:
                    good = outer.case["expected_source_used"] == "web_search"
                    return schema(grade="good" if good else "weak", supporting_chunk_ids=[])
                docs = outer.case["offline"]["retrieved_chunks"]
                supporting = [str(c["metadata"].get("chunk_id")) for c in docs if c.get("support", True)]
                good = outer.case["expected_source_used"] == "private_kb" and bool(supporting)
                return schema(grade="good" if good else "weak", supporting_chunk_ids=supporting if good else [])

        return Structured()

    def invoke(self, prompt: str):
        prompt = _prompt_text(prompt)
        if "Rewrite the latest message" in prompt and self.case.get("retrieval_query"):
            return AIMessage(content=self.case["retrieval_query"])
        if "Rewrite the question for retrieval" in prompt:
            return AIMessage(content=self.case.get("retrieval_query", self.case["question"]))
        return AIMessage(content=self.case.get("offline", {}).get("mock_answer", ""))


class _FixtureWebSearch:
    def __init__(self, case: dict[str, Any]):
        self.case = case

    def invoke(self, query: dict[str, str]):
        return {"answer": self.case.get("offline", {}).get("mock_answer", ""), "results": []}


class _RecordingRetriever:
    def __init__(self, inner, record: list[dict[str, Any]]):
        self.inner, self.record = inner, record

    def invoke(self, query: str):
        docs = self.inner.invoke(query)
        self.record[:] = [
            {"source": str(d.metadata.get("source_filename") or d.metadata.get("source") or "unknown"),
             "page": d.metadata.get("page"), "content_type": d.metadata.get("content_type"),
             "chunk_id": d.metadata.get("chunk_id")}
            for d in docs
        ]
        return docs


def _safe_configuration(mode: str) -> dict[str, Any]:
    settings = get_settings()
    return {
        "mode": mode,
        "app_env": settings.app_env,
        "index_name": settings.pinecone_index_name if mode == "live" else None,
        "namespace": settings.pinecone_namespace if mode == "live" else None,
        "top_k": settings.top_k,
        "retrieval_candidate_k": settings.retrieval_candidate_k,
        "embedding_model": settings.embedding_model,
        "answer_model": settings.openrouter_model if mode == "live" else "fixture-mock",
        "vision_model": settings.vision_model or None,
        "web_search_enabled": bool(settings.tavily_api_key) if mode == "live" else False,
    }


def _citations_as_source_pages(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for citation in citations:
        if citation.get("source"):
            output.append({"source": citation["source"], "page": citation.get("page")})
            continue
        title = str(citation.get("title") or "")
        match = re.search(r"(.+?)\s*[—-]\s*page\s+(\d+)", title, flags=re.I)
        output.append({"source": match.group(1).strip() if match else title, "page": int(match.group(2)) if match else citation.get("page")})
    return output


def _one_case(case: dict[str, Any], mode: str) -> dict[str, Any]:
    ranked: list[dict[str, Any]] = []
    original_retriever = workflow.get_retriever
    original_llm = workflow.llm
    original_web_search = workflow.web_search_tool
    try:
        if mode == "offline":
            workflow.get_retriever = lambda: _FixtureRetriever(case, ranked)
            workflow.llm = lambda: _FixtureLLM(case)
            workflow.web_search_tool = lambda: _FixtureWebSearch(case)
        else:
            workflow.get_retriever = lambda: _RecordingRetriever(original_retriever(), ranked)
        graph = workflow.build_graph()
        messages = []
        for turn in case.get("history", []):
            role, content = turn.get("role"), turn.get("content", "")
            messages.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
        messages.append(HumanMessage(content=case["question"]))
        result = graph.invoke({
            "question": case["question"], "messages": messages,
            "current_query": case["question"], "kb_docs": [], "web_results": "",
            "kb_grade": "", "web_grade": "", "answer": "", "source_used": "",
            "retry_count": 0, "trace": [], "citations": [],
        })
        expected = case["expected_evidence"]
        retrieval = retrieval_metrics(expected, ranked, k=5)
        citations = _citations_as_source_pages(result.get("citations", []))
        citation = citation_metrics(expected, citations)
        answer = str(result.get("answer", ""))
        abstained = result.get("source_used") == "insufficient_evidence"
        expected_source = case["expected_source_used"]
        answer_score = answer_fact_score(case["expected_facts"], answer)
        checks = {
            "source_correct": result.get("source_used") == expected_source,
            "answer_facts_all_found": answer_score == 1.0 if answer_score is not None else None,
            "abstention_correct": abstained == case["should_abstain"],
            "retrieval_recall_positive": retrieval["recall_at_5"] is None or retrieval["recall_at_5"] > 0,
        }
        return {
            "case_id": case["id"], "retrieval": retrieval, "citations": citation,
            "answer_correctness": answer_score,
            "source_correct": checks["source_correct"],
            "should_abstain": case["should_abstain"], "abstained": abstained,
            "abstention_correct": checks["abstention_correct"],
            "retrieved": ranked, "cited": citations,
            "failed_checks": [name for name, value in checks.items() if value is False],
        }
    finally:
        workflow.get_retriever = original_retriever
        workflow.llm = original_llm
        workflow.web_search_tool = original_web_search


def run_evaluation(dataset: dict[str, Any], mode: str = "offline") -> dict[str, Any]:
    if mode not in {"offline", "live"}:
        raise ValueError("mode must be offline or live")
    restore_ensure_index = None
    if mode == "live":
        settings = get_settings()
        if not settings.pinecone_api_key:
            raise RuntimeError("Live evaluation requires PINECONE_API_KEY")
        # Refuse the implicit index-creation path used by app startup. The workflow
        # will receive this already-existing index handle for the duration of eval.
        from pinecone import Pinecone
        from app.rag import vectorstore
        pc = Pinecone(api_key=settings.pinecone_api_key)
        names = {entry["name"] for entry in pc.list_indexes()}
        if settings.pinecone_index_name not in names:
            raise RuntimeError("Live evaluation target index does not exist; refusing to create it")
        restore_ensure_index = vectorstore.ensure_index
        vectorstore.ensure_index = lambda: pc.Index(settings.pinecone_index_name)
    try:
        cases = [_one_case(case, mode) for case in dataset["cases"]]
    finally:
        if restore_ensure_index is not None:
            from app.rag import vectorstore
            vectorstore.ensure_index = restore_ensure_index
    totals: dict[str, Any] = {
        "answer_correctness": _mean([case["answer_correctness"] for case in cases]),
        "source_accuracy": _mean([1.0 if case["source_correct"] else 0.0 for case in cases]),
        **aggregate_abstention(cases),
    }
    for name in ("recall_at_5", "mrr"):
        totals[name] = _mean([case["retrieval"][name] for case in cases])
    for name in ("citation_precision", "citation_coverage"):
        totals[name] = _mean([case["citations"][name] for case in cases])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset["dataset_version"],
        "configuration": _safe_configuration(mode),
        "summary": totals,
        "cases": cases,
        "failed_case_ids": sorted({case["case_id"] for case in cases if case["failed_checks"]}),
        "limitations": [
            "Offline answer grading and retrieval fixtures are deterministic mocks, not model-quality measurements."
        ] if mode == "offline" else [
            "Live results depend on provider/model nondeterminism and the configured active-generation registry.",
            "This runner performs read-only retrieval; it does not ingest, upsert, or delete vectors.",
        ],
    }


def _mean(values: list[float | None]) -> float | None:
    measured = [value for value in values if value is not None]
    return sum(measured) / len(measured) if measured else None


def save_report(report: dict[str, Any], directory: str | Path) -> Path:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output / f"rag_eval_{stamp}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
