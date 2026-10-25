from typing import List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langgraph.graph import MessagesState

class RouteDecision(BaseModel):
    route: Literal["kb", "direct"] = Field(description="kb for IT/support questions; direct for greetings/simple chat")

class EvidenceGrade(BaseModel):
    grade: Literal["good", "weak"] = Field(description="Whether evidence is sufficient to answer")
    supporting_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Exact chunk IDs that directly support the answer",
    )

class AgentState(MessagesState, total=False):
    question: str
    current_query: str
    kb_docs: List[Document]
    kb_supporting_chunk_ids: List[str]
    web_results: str
    kb_grade: str
    web_grade: str
    answer: str
    source_used: str
    retry_count: int
    trace: List[str]
    citations: List[dict]
    selected_route: str
    retrieval_results: List[dict]
    stage_timings_ms: dict[str, float]
    query_rewrite_count: int
