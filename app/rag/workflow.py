import logging
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph

from app.core.config import get_settings
from app.rag.state import AgentState, EvidenceGrade, RouteDecision
from app.rag.vectorstore import get_retriever


logger = logging.getLogger(__name__)
settings = get_settings()

_llm: ChatOpenAI | None = None
_web_search: TavilySearch | None = None


def llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        if not settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing")
        _llm = ChatOpenAI(
            model=settings.openrouter_model,
            base_url=settings.openrouter_base_url,
            temperature=0,
            api_key=settings.openrouter_api_key,
        )
    return _llm


def web_search_tool() -> TavilySearch:
    global _web_search
    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")
        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )
    return _web_search


def add_trace(state: AgentState, message: str) -> list[str]:
    return [*state.get("trace", []), message]


def conversation_history(state: AgentState) -> str:
    """Format prior turns separately from retrieved knowledge documents."""
    messages = state.get("messages", [])
    prior = messages[:-1] if messages and isinstance(messages[-1], HumanMessage) else messages
    prior = prior[-12:]
    return "\n".join(
        f"{('User' if isinstance(message, HumanMessage) else 'Assistant')}: {message.content}"
        for message in prior
        if isinstance(message.content, str) and message.content.strip()
    )


def resolve_query(state: AgentState) -> dict[str, object]:
    history = conversation_history(state)
    if not history:
        return {"current_query": state["question"]}
    standalone = llm().invoke(
        f"""
Rewrite the latest user message as a standalone search query for an IT support assistant.
Use the conversation only to resolve references, pronouns, omitted subjects, and clarifications.
If the latest message is already standalone or is casual chat, return it unchanged.
Do not answer. Return only the rewritten query.

Conversation history:
{history}
Latest user message (preserve it as the user's original question):
{state['question']}
"""
    ).content
    query = standalone.strip() if isinstance(standalone, str) else ""
    if not query:
        query = state["question"]
    return {
        "current_query": query,
        "trace": add_trace(state, "Follow-up context resolved"),
    }


def route_question(state: AgentState) -> dict[str, object]:
    router = llm().with_structured_output(RouteDecision, method="json_mode")
    decision = router.invoke(
        f"""
You route messages for an enterprise IT support assistant.
Use kb for questions about company IT policies, VPN, password reset, MFA, laptop setup,
software access, security, email, devices, troubleshooting, or technology support.
Use direct only for greetings, thanks, or casual chat that needs no company knowledge.
Question: {state["current_query"]}
Return valid JSON like {{"route":"kb"}}.
"""
    )
    logger.info("Question routed to %s", decision.route)
    return {
        "source_used": decision.route,
        "trace": add_trace(state, f"Router → {decision.route.upper()}"),
    }


def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    return "retrieve_kb" if state["source_used"] == "kb" else "direct_answer"


def retrieve_kb(state: AgentState) -> dict[str, object]:
    docs = get_retriever().invoke(state["current_query"])
    logger.info("Retrieved %d private KB chunks", len(docs))
    return {
        "kb_docs": docs,
        "trace": add_trace(state, f"Private KB retrieval → {len(docs)} chunks"),
    }


def grade_kb(state: AgentState) -> dict[str, object]:
    grader = llm().with_structured_output(EvidenceGrade, method="json_mode")
    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in state["kb_docs"]
    )
    grade = grader.invoke(
        f"""
You grade evidence for an enterprise IT support assistant.
Question: {state["current_query"]}
Private company KB evidence:
{context}
Return good only if the evidence is sufficient to answer confidently and specifically.
Otherwise return weak. JSON: {{"grade":"good"}} or {{"grade":"weak"}}.
"""
    )
    logger.info("Private KB evidence graded %s", grade.grade)
    return {
        "kb_grade": grade.grade,
        "trace": add_trace(state, f"KB evidence grade → {grade.grade.upper()}"),
    }


def after_kb(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
    return "generate_from_kb" if state["kb_grade"] == "good" else "search_web"


def search_web(state: AgentState) -> dict[str, object]:
    result = web_search_tool().invoke({"query": state["current_query"]})
    lines: list[str] = []
    citations: list[dict[str, str]] = []

    if isinstance(result, dict):
        answer = result.get("answer")
        if answer:
            lines.append(f"Search answer: {answer}")
        for item in result.get("results", []):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "")
            url = str(item.get("url") or "")
            content = str(item.get("content") or "")
            lines.append(f"Title: {title}\nURL: {url}\nContent: {content}")
            if url:
                citations.append(
                    {"title": title or url, "url": url, "type": "web"}
                )
    else:
        lines.append(str(result))

    web_results = "\n\n".join(lines)
    logger.info("Tavily search returned %d characters", len(web_results))
    return {
        "web_results": web_results,
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(state, "Web fallback → Tavily search"),
    }


def grade_web(state: AgentState) -> dict[str, object]:
    grader = llm().with_structured_output(EvidenceGrade, method="json_mode")
    grade = grader.invoke(
        f"""
Question: {state["current_query"]}
Web evidence:
{state["web_results"]}
Return good if the evidence is sufficient and directly relevant; otherwise weak.
Return valid JSON like {{"grade":"good"}}.
"""
    )
    logger.info("Web evidence graded %s", grade.grade)
    return {
        "web_grade": grade.grade,
        "trace": add_trace(state, f"Web evidence grade → {grade.grade.upper()}"),
    }


def after_web(
    state: AgentState,
) -> Literal["generate_from_web", "rewrite_query", "insufficient"]:
    if state["web_grade"] == "good":
        return "generate_from_web"
    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"
    return "insufficient"


def rewrite_query(state: AgentState) -> dict[str, object]:
    rewritten = llm().invoke(
        f"""
Rewrite this IT support question for better private knowledge retrieval and vendor web search.
Preserve intent, add useful technical keywords, do not answer, return only the query.
Question: {state["current_query"]}
"""
    ).content
    if not isinstance(rewritten, str) or not rewritten.strip():
        raise ValueError("The model returned an empty rewritten search query")
    rewritten = rewritten.strip()
    logger.info("Rewrote search query")
    return {
        "current_query": rewritten,
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(state, f"Query rewrite → {rewritten}"),
    }


def generate_from_kb(state: AgentState) -> dict[str, object]:
    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in state["kb_docs"]
    )
    history = conversation_history(state) or "No prior conversation."
    answer = llm().invoke(
        f"""
You are an enterprise IT support copilot. Answer ONLY from the private company KB below.
Be concise, actionable, and safe. If steps are present, present them clearly.
Do not invent policy details. Mention that the answer is based on the company's private knowledge base.
Conversation history (for continuity; not evidence):
{history}

Current user question:
{state["question"]}

Private KB:
{context}
"""
    ).content

    citations: list[dict[str, str]] = []
    seen: set[str] = set()
    for doc in state["kb_docs"]:
        source = str(doc.metadata.get("source") or "Private KB")
        if source not in seen:
            seen.add(source)
            citations.append(
                {
                    "title": source.replace("\\", "/").rsplit("/", 1)[-1],
                    "url": "",
                    "type": "private_kb",
                }
            )

    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
        "source_used": "private_kb",
        "citations": citations,
        "trace": add_trace(state, "Answer generation → PRIVATE KB"),
    }


def generate_from_web(state: AgentState) -> dict[str, object]:
    history = conversation_history(state) or "No prior conversation."
    answer = llm().invoke(
        f"""
You are an enterprise IT support copilot. The private company KB was insufficient.
Answer ONLY from the web evidence below. Clearly say this is external web information
and may need IT validation before changing company-managed systems.
Conversation history (for continuity; not evidence):
{history}

Current user question:
{state["question"]}

Web evidence:
{state["web_results"]}
"""
    ).content
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
        "source_used": "web_search",
        "trace": add_trace(state, "Answer generation → WEB SEARCH"),
    }


def direct_answer(state: AgentState) -> dict[str, object]:
    answer = llm().invoke(
        f"""Respond briefly and naturally, maintaining continuity with the conversation.
Conversation history:
{conversation_history(state) or 'No prior conversation.'}
Current user message: {state['question']}"""
    ).content
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
        "source_used": "direct",
        "trace": add_trace(state, "Direct response → no retrieval"),
    }


def insufficient(state: AgentState) -> dict[str, object]:
    return {
        "answer": (
            "I couldn't find enough reliable evidence in the company knowledge base "
            "or external search to answer confidently. Please contact the IT help "
            "desk or provide more details."
        ),
        "messages": [AIMessage(content=(
            "I couldn't find enough reliable evidence in the company knowledge base "
            "or external search to answer confidently. Please contact the IT help "
            "desk or provide more details."
        ))],
        "source_used": "insufficient_evidence",
        "trace": add_trace(state, "Stopped → insufficient reliable evidence"),
    }


def build_graph(checkpointer=None):
    graph = StateGraph(AgentState)
    for name, node in {
        "resolve_query": resolve_query,
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }.items():
        graph.add_node(name, node)

    graph.add_edge(START, "resolve_query")
    graph.add_edge("resolve_query", "route_question")
    graph.add_conditional_edges(
        "route_question",
        route_after_router,
        {"retrieve_kb": "retrieve_kb", "direct_answer": "direct_answer"},
    )
    graph.add_edge("retrieve_kb", "grade_kb")
    graph.add_conditional_edges(
        "grade_kb",
        after_kb,
        {
            "generate_from_kb": "generate_from_kb",
            "search_web": "search_web",
        },
    )
    graph.add_edge("search_web", "grade_web")
    graph.add_conditional_edges(
        "grade_web",
        after_web,
        {
            "generate_from_web": "generate_from_web",
            "rewrite_query": "rewrite_query",
            "insufficient": "insufficient",
        },
    )
    graph.add_edge("rewrite_query", "retrieve_kb")
    for terminal_node in (
        "generate_from_kb",
        "generate_from_web",
        "direct_answer",
        "insufficient",
    ):
        graph.add_edge(terminal_node, END)

    return graph.compile(checkpointer=checkpointer)


agent_graph = build_graph()


def ask(question: str, thread_id: str | None = None, graph=None) -> AgentState:
    if not question.strip():
        raise ValueError("Question must not be empty")

    initial: AgentState = {
        "question": question,
        "messages": [HumanMessage(content=question)],
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }
    active_graph = graph or agent_graph
    config = {"configurable": {"thread_id": thread_id}} if thread_id else None
    return active_graph.invoke(initial, config=config)
