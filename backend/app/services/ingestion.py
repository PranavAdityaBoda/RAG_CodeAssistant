"""
Ingestion pipeline: clone, walk, chunk, embed.
"""
import uuid

from app.core.logging import get_logger
from app.services import cloner, job_tracker, vector_store
from app.services.chunker import chunk_file
from app.services.cleanup import purge_old_jobs, purge_previous_jobs_for_url
from app.services.walker import walk_repo

logger = get_logger(__name__)


def start_ingestion(repo_url: str) -> str:
    """Creates a job record and returns the job_id. Work runs in the background."""
    job_id = uuid.uuid4().hex[:12]
    job_tracker.create_job(job_id, repo_url)
    return job_id


def run_ingestion_bg(job_id: str, repo_url: str) -> None:
    """Background worker. Updates job status at each stage."""
    repo_path = None
    try:
        purge_previous_jobs_for_url(repo_url)

        job_tracker.update_job(job_id, status="cloning")
        repo_path = cloner.clone_repo(repo_url, job_id=job_id)

        job_tracker.update_job(job_id, status="walking")
        files = walk_repo(repo_path)
        job_tracker.update_job(job_id, files_discovered=len(files))

        job_tracker.update_job(job_id, status="chunking")
        chunks = []
        for f in files:
            chunks.extend(chunk_file(f))
        job_tracker.update_job(job_id, files_chunked=len(files), chunks_created=len(chunks))

        job_tracker.update_job(job_id, status="embedding")
        vector_store.store_chunks(job_id, chunks)

        job_tracker.update_job(
            job_id,
            status="done",
            result_files=[
                {"path": f.relative_path, "extension": f.extension,
                 "size_bytes": f.size_bytes, "language": f.language}
                for f in files
            ],
            result_chunks=[
                {"chunk_id": c.chunk_id, "file_path": c.file_path,
                 "symbol_name": c.symbol_name, "chunk_type": c.chunk_type,
                 "start_line": c.start_line, "end_line": c.end_line,
                 "language": c.language, "preview": c.text[:200]}
                for c in chunks
            ],
        )
        logger.info("Ingestion done: job=%s files=%d chunks=%d", job_id, len(files), len(chunks))
        purge_old_jobs()

    except Exception as exc:
        logger.exception("Ingestion failed for job %s", job_id)
        job_tracker.update_job(job_id, status="failed", error=str(exc))

    finally:
        if repo_path is not None:
            cloner.delete_clone(repo_path)
