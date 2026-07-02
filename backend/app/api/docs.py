"""
HTTP routes for doc generation and codebase Q&A.

POST /api/docs/generate , starts doc generation as a background task,
                           returns a doc_job_id immediately (no more timeout).
GET  /api/docs/generate/status/{doc_job_id}, poll until status is done/failed.
POST /api/docs/qa       , synchronous Q&A (fast enough not to need background).
GET  /api/docs/usage    , Groq free-tier usage stats.
"""
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.core.logging import get_logger
from app.models.schemas import (
    DocGenerateRequest,
    DocGenerateStarted,
    DocJobStatus,
    QARequest,
    QAResponse,
)
from app.services import doc_generator, job_tracker, rag
from app.services.job_tracker import get_job

logger = get_logger(__name__)
router = APIRouter(prefix="/api/docs", tags=["docs"])


def _assert_job_ready(job_id: str) -> None:
    """Shared guard: raises 404 if job doesn't exist, 400 if not done."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"No job found with id '{job_id}'. Run POST /api/ingest first.",
        )
    if job.status != "done":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Job '{job_id}' is not ready (status: {job.status}). "
                "Wait for ingestion to complete."
            ),
        )


def _run_doc_generation_bg(
    doc_job_id: str,
    ingest_job_id: str,
    repo_url: str,
    doc_type: str,
) -> None:
    """
    Background worker: runs the full doc generation pipeline and updates
    the doc job state when done. Errors are caught and stored so the
    frontend can surface them via the polling endpoint.
    """
    try:
        result = doc_generator.generate_docs(
            job_id=ingest_job_id,
            repo_url=repo_url,
            doc_type=doc_type,
        )
        job_tracker.update_doc_job(
            doc_job_id,
            status="done",
            content=result["content"],
            files_summarised=result["files_summarised"],
            llm_calls_used=result["llm_calls_used"],
        )
        logger.info("Doc job %s finished successfully", doc_job_id)
    except Exception as exc:
        logger.exception("Doc job %s failed", doc_job_id)
        job_tracker.update_doc_job(
            doc_job_id,
            status="failed",
            error=str(exc),
        )


@router.post("/generate", response_model=DocGenerateStarted)
def generate_docs(
    request: DocGenerateRequest,
    background_tasks: BackgroundTasks,
) -> DocGenerateStarted:
    """
    Starts doc generation in the background and returns a doc_job_id immediately.

    Poll GET /api/docs/generate/status/{doc_job_id} for progress.
    This avoids the frontend HTTP timeout that occurred when blocking on a
    long Groq run with rate-limit retries.
    """
    _assert_job_ready(request.job_id)
    job = get_job(request.job_id)

    doc_job_id = uuid.uuid4().hex[:12]
    job_tracker.create_doc_job(doc_job_id, request.job_id, request.doc_type)

    background_tasks.add_task(
        _run_doc_generation_bg,
        doc_job_id=doc_job_id,
        ingest_job_id=request.job_id,
        repo_url=job.repo_url,
        doc_type=request.doc_type,
    )

    logger.info(
        "Doc job %s started (ingest=%s, type=%s)",
        doc_job_id, request.job_id, request.doc_type,
    )
    return DocGenerateStarted(
        doc_job_id=doc_job_id,
        ingest_job_id=request.job_id,
        doc_type=request.doc_type,
        message="Doc generation started. Poll /api/docs/generate/status/{doc_job_id}.",
    )


@router.get("/generate/status/{doc_job_id}", response_model=DocJobStatus)
def get_doc_job_status(doc_job_id: str) -> DocJobStatus:
    """
    Returns the current state of a doc generation job.
    Status values: running | done | failed.
    """
    state = job_tracker.get_doc_job(doc_job_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"No doc job found with id '{doc_job_id}'.",
        )
    return DocJobStatus(**state)


@router.post("/qa", response_model=QAResponse)
def qa(request: QARequest) -> QAResponse:
    """
    Answers a natural-language question about an ingested codebase.
    Synchronous, Q&A is a single Groq call, fast enough not to need a
    background task.
    """
    _assert_job_ready(request.job_id)

    try:
        result = rag.answer_question(
            job_id=request.job_id,
            question=request.question,
            history=request.history,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Q&A failed: {exc}") from exc

    return QAResponse(
        job_id=request.job_id,
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
    )


@router.get("/usage")
def get_llm_usage() -> dict:
    """Returns current Groq request usage against the free-tier limits."""
    try:
        from app.services.llm_client import get_llm_client
        return get_llm_client().usage
    except ValueError:
        return {"error": "LLM client not initialised. Check GROQ_API_KEY."}
