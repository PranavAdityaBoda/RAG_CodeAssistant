"""
HTTP routes for repo ingestion.

"""
from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.core.logging import get_logger
from app.models.schemas import IngestRequest, IngestResponse, IngestResult, JobStatus, FileInfo, ChunkInfo
from app.services import job_tracker
from app.services.cloner import CloneError
from app.services.ingestion import run_ingestion_bg, start_ingestion

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["ingestion"])


@router.post("/ingest", response_model=IngestResponse)
def ingest_repository(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    """
    Starts ingestion as a background task and returns a job_id immediately.
    Poll GET /api/jobs/{job_id} for status.
    When status is 'done', fetch GET /api/jobs/{job_id}/result for files/chunks.
    """
    try:
        job_id = start_ingestion(request.repo_url)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    background_tasks.add_task(run_ingestion_bg, job_id, request.repo_url)

    return IngestResponse(
        job_id=job_id,
        repo_url=request.repo_url,
        status="queued",
        message="Ingestion started. Poll /api/jobs/{job_id} for status.",
    )


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str) -> JobStatus:
    state = job_tracker.get_job(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"No job found with id '{job_id}'")
    return JobStatus(
        job_id=state.job_id,
        repo_url=state.repo_url,
        status=state.status,
        files_discovered=state.files_discovered,
        files_chunked=state.files_chunked,
        chunks_created=state.chunks_created,
        error=state.error,
    )


@router.get("/jobs/{job_id}/result", response_model=IngestResult)
def get_job_result(job_id: str) -> IngestResult:
    """
    Returns the full file and chunk list for a completed ingestion job.
    Returns 404 if not found, 400 if the job is not yet done.
    """
    state = job_tracker.get_job(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"No job found with id '{job_id}'")
    if state.status != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not complete yet (status: {state.status})",
        )

    # Retrieve files and chunks stored by the background worker
    files = [FileInfo(**f) for f in (state.result_files or [])]
    chunks = [ChunkInfo(**c) for c in (state.result_chunks or [])]

    return IngestResult(
        job_id=state.job_id,
        repo_url=state.repo_url,
        status=state.status,
        files=files,
        chunks=chunks,
    )
