"""
Tracks ingestion job status.

In-memory dict for fast reads during a run, backed by SQLite so job history
survives a backend restart.
"""
import json
import sqlite3
import threading
from dataclasses import asdict, dataclass, field

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_jobs: dict[str, "JobState"] = {}


@dataclass
class JobState:
    job_id: str
    repo_url: str
    status: str = "queued"
    files_discovered: int = 0
    files_chunked: int = 0
    chunks_created: int = 0
    error: str | None = None
    # Stored as JSON blobs in SQLite, populated when status reaches 'done'
    result_files: list | None = None
    result_chunks: list | None = None


def _get_db() -> sqlite3.Connection:
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.sqlite_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            repo_url TEXT NOT NULL,
            status TEXT NOT NULL,
            files_discovered INTEGER DEFAULT 0,
            files_chunked INTEGER DEFAULT 0,
            chunks_created INTEGER DEFAULT 0,
            error TEXT,
            result_files TEXT,
            result_chunks TEXT
        )
        """
    )
    for col, typedef in [("result_files", "TEXT"), ("result_chunks", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {typedef}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
    return conn


def _persist(state: JobState) -> None:
    conn = _get_db()
    with conn:
        conn.execute(
            """
            INSERT INTO jobs (job_id, repo_url, status, files_discovered,
                               files_chunked, chunks_created, error,
                               result_files, result_chunks)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                status=excluded.status,
                files_discovered=excluded.files_discovered,
                files_chunked=excluded.files_chunked,
                chunks_created=excluded.chunks_created,
                error=excluded.error,
                result_files=excluded.result_files,
                result_chunks=excluded.result_chunks
            """,
            (
                state.job_id, state.repo_url, state.status,
                state.files_discovered, state.files_chunked,
                state.chunks_created, state.error,
                json.dumps(state.result_files) if state.result_files is not None else None,
                json.dumps(state.result_chunks) if state.result_chunks is not None else None,
            ),
        )
    conn.close()


def create_job(job_id: str, repo_url: str) -> JobState:
    state = JobState(job_id=job_id, repo_url=repo_url, status="queued")
    with _lock:
        _jobs[job_id] = state
    _persist(state)
    return state


def update_job(job_id: str, **fields) -> JobState:
    with _lock:
        state = _jobs.get(job_id)
        if state is None:
            raise KeyError(f"Unknown job_id: {job_id}")
        for key, value in fields.items():
            setattr(state, key, value)
    _persist(state)
    return state


def get_job(job_id: str) -> JobState | None:
    with _lock:
        state = _jobs.get(job_id)
    if state is not None:
        return state

    conn = _get_db()
    row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    conn.close()
    if row is None:
        return None

    columns = ["job_id", "repo_url", "status", "files_discovered",
               "files_chunked", "chunks_created", "error",
               "result_files", "result_chunks"]
    data = dict(zip(columns, row))
    # Deserialise JSON blobs back to lists
    for key in ("result_files", "result_chunks"):
        raw = data.get(key)
        data[key] = json.loads(raw) if raw else None

    return JobState(**data)


# ── Doc generation job tracking ───────────────────────────────────────────
# Doc generation runs as a background task and can take minutes on a free-tier Groq account. 
# A separate lightweight tracker lets the frontend poll for completion without holding an HTTP connection open.

_doc_jobs: dict[str, dict] = {}


def create_doc_job(doc_job_id: str, ingest_job_id: str, doc_type: str) -> dict:
    state = {
        "doc_job_id": doc_job_id,
        "ingest_job_id": ingest_job_id,
        "doc_type": doc_type,
        "status": "running",      # running | done | failed
        "content": None,
        "files_summarised": 0,
        "llm_calls_used": 0,
        "error": None,
    }
    with _lock:
        _doc_jobs[doc_job_id] = state
    return state


def update_doc_job(doc_job_id: str, **fields) -> None:
    with _lock:
        if doc_job_id in _doc_jobs:
            _doc_jobs[doc_job_id].update(fields)


def get_doc_job(doc_job_id: str) -> dict | None:
    with _lock:
        return dict(_doc_jobs.get(doc_job_id, {})) or None
