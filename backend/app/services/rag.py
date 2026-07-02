"""
RAG Q&A, retrieval + Groq generation over an ingested repo.

Query expansion runs before retrieval to catch vocabulary mismatches
(user says "auth flow", code says "token_validate"). History capped at
3 turns so we don't blow the TPM limit on long conversations.
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.core.config import settings
from app.core.logging import get_logger
from app.services import vector_store
from app.services.llm_client import get_llm_client
from app.services.router import route_task

logger = get_logger(__name__)


_SYS_PROMPT = """\
You are a senior software engineer doing a code review walkthrough.
You have specific code chunks from the repo as context.

Rules:
1. Only answer questions about the ingested codebase. If the question is
   unrelated to the code in the context (e.g. general knowledge, other repos,
   personal advice), say: "I can only answer questions about the ingested repo."
2. Answer directly, name exact files, functions, classes.
3. Explain the *why*, not just the *what*.
4. For flow questions, trace step by step through the actual files.
5. If context is thin, say what you can confirm and what's missing.

Prose only, no bullet padding. Use `code spans` for symbols and paths.
"""

# Distance threshold, Chroma cosine distance above this means the question
# probably has nothing to do with the ingested repo
_MAX_RELEVANT_DISTANCE = 1.2

_EXPAND_SYS = """\
Given a question about a codebase, output 2 alternative search queries.
Exactly 2 lines, one query each, no numbering, no explanation.
"""


def _expand(q: str) -> list[str]:
    try:
        client = get_llm_client()
        raw = client.complete(
            model_tier="fast",
            system_prompt=_EXPAND_SYS,
            user_prompt=f"Question: {q}",
            temperature=0.4,
        )
        return [l.strip() for l in raw.strip().splitlines() if l.strip()][:2]
    except Exception as exc:
        logger.warning("Query expansion failed: %s", exc)
        return []


def _retrieve(job_id: str, question: str, extras: list[str], k: int) -> list[dict]:
    seen: dict[str, dict] = {}
    for q in [question] + extras:
        for chunk in vector_store.query_chunks(job_id=job_id, query=q, top_k=k):
            cid = chunk["metadata"].get("file_path", "") + str(chunk["metadata"].get("start_line", ""))
            if cid not in seen or chunk.get("distance", 1) < seen[cid].get("distance", 1):
                seen[cid] = chunk
    return sorted(seen.values(), key=lambda c: c.get("distance", 1))[:k + 2]


def _fmt_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        header = f"[{i}] `{m.get('file_path', '?')}` `{m.get('symbol_name') or m.get('chunk_type', 'block')}` (lines {m.get('start_line','?')}-{m.get('end_line','?')})"
        parts.append(f"{header}\n{c['text'][:600]}")
    return "\n\n---\n\n".join(parts)


def _build_msgs(history: list[dict], ctx: str, question: str) -> list:
    msgs = [SystemMessage(content=f"{_SYS_PROMPT}\n\n--- Code Context ---\n{ctx}")]
    for turn in history[-6:]:
        role, content = turn.get("role"), turn.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    msgs.append(HumanMessage(content=question))
    return msgs


def answer_question(job_id: str, question: str, history: list[dict] | None = None) -> dict:
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY not set.")

    history = history or []
    extras = _expand(question)
    logger.info("Q&A: %r, %d expansions", question[:60], len(extras))

    chunks = _retrieve(job_id, question, extras, settings.qa_top_k)
    if not chunks:
        return {"answer": "No relevant code found. Check the repo was ingested OK.", "sources": []}

    # If even the best chunk is far away, the question is probably off-topic
    best_distance = chunks[0].get("distance", 0)
    if best_distance > _MAX_RELEVANT_DISTANCE:
        return {
            "answer": "I can only answer questions about the ingested repo. "
                      "This question doesn't seem related to the codebase, try asking "
                      "about a specific file, function or feature from the repo.",
            "sources": [],
        }

    tier = route_task("qa_answer")
    model = settings.groq_fast_model if tier == "fast" else settings.groq_reasoning_model
    llm = ChatGroq(api_key=settings.groq_api_key, model_name=model, temperature=0.2)

    logger.info("Calling %s with %d chunks", model, len(chunks))
    resp = llm.invoke(_build_msgs(history, _fmt_context(chunks), question))
    answer = resp.content if hasattr(resp, "content") else str(resp)

    sources = [
        {
            "file_path":   c["metadata"].get("file_path", ""),
            "symbol_name": c["metadata"].get("symbol_name", ""),
            "chunk_type":  c["metadata"].get("chunk_type", ""),
            "start_line":  c["metadata"].get("start_line", 0),
            "end_line":    c["metadata"].get("end_line", 0),
            "distance":    round(c.get("distance", 0), 4),
        }
        for c in chunks
    ]
    return {"answer": answer, "sources": sources}
