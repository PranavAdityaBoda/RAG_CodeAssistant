"""
Tiny HTTP client wrapping calls to the FastAPI backend.

Keeping all requests.* calls in one module means the Streamlit app code
itself never touches URLs or response parsing directly, and the base URL
only needs to change in one place.
"""
import os

import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
_TIMEOUT_SECONDS = 300  # ingestion can take a while on larger repos


class BackendError(Exception):
    """Raised when the backend returns an error response."""


def ingest_repository(repo_url: str) -> dict:
    """
    Starts ingestion and returns immediately with job_id + status='queued'.
    Poll get_ingest_status() until done, then call get_ingest_result().
    """
    response = requests.post(
        f"{BACKEND_URL}/api/ingest",
        json={"repo_url": repo_url},
        timeout=30,
    )
    if response.status_code != 200:
        detail = _extract_error_detail(response)
        raise BackendError(detail)
    return response.json()


def get_ingest_status(job_id: str) -> dict:
    """Polls GET /api/jobs/{job_id} for ingestion status."""
    response = requests.get(f"{BACKEND_URL}/api/jobs/{job_id}", timeout=10)
    if response.status_code != 200:
        raise BackendError(_extract_error_detail(response))
    return response.json()


def get_ingest_result(job_id: str) -> dict:
    """Fetches the full file/chunk result for a completed ingestion job."""
    response = requests.get(f"{BACKEND_URL}/api/jobs/{job_id}/result", timeout=10)
    if response.status_code != 200:
        raise BackendError(_extract_error_detail(response))
    return response.json()


def get_job_status(job_id: str) -> dict:
    response = requests.get(f"{BACKEND_URL}/api/jobs/{job_id}", timeout=30)
    if response.status_code != 200:
        detail = _extract_error_detail(response)
        raise BackendError(detail)
    return response.json()


def start_doc_generation(job_id: str, doc_type: str = "api") -> dict:
    """
    Starts doc generation as a background task.
    Returns immediately with a doc_job_id to poll.
    """
    response = requests.post(
        f"{BACKEND_URL}/api/docs/generate",
        json={"job_id": job_id, "doc_type": doc_type},
        timeout=30,   # just the POST to start, fast, no LLM work yet
    )
    if response.status_code != 200:
        detail = _extract_error_detail(response)
        raise BackendError(detail)
    return response.json()


def poll_doc_job(doc_job_id: str) -> dict:
    """
    Returns the current state of a background doc generation job.
    Status: running | done | failed.
    """
    response = requests.get(
        f"{BACKEND_URL}/api/docs/generate/status/{doc_job_id}",
        timeout=10,
    )
    if response.status_code != 200:
        detail = _extract_error_detail(response)
        raise BackendError(detail)
    return response.json()


def ask_question(job_id: str, question: str, history: list | None = None) -> dict:
    response = requests.post(
        f"{BACKEND_URL}/api/docs/qa",
        json={"job_id": job_id, "question": question, "history": history or []},
        timeout=60,   # single Q&A call, 60s is generous enough
    )
    if response.status_code != 200:
        detail = _extract_error_detail(response)
        raise BackendError(detail)
    return response.json()


def get_llm_usage() -> dict:
    try:
        response = requests.get(f"{BACKEND_URL}/api/docs/usage", timeout=10)
        return response.json() if response.status_code == 200 else {}
    except requests.RequestException:
        return {}


def check_backend_health() -> bool:
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


# ── GitHub OAuth + PR Agent ───────────────────────────────────────────────

def get_github_login_url() -> str:
    """Returns the GitHub OAuth authorisation URL to open in a new tab."""
    response = requests.get(f"{BACKEND_URL}/api/github/login", timeout=10)
    if response.status_code != 200:
        raise BackendError(_extract_error_detail(response))
    return response.json()["auth_url"]


def get_github_status(token: str) -> dict:
    response = requests.get(
        f"{BACKEND_URL}/api/github/status",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    if response.status_code != 200:
        return {"connected": False}
    return response.json()


def create_pull_request(
    github_token: str,
    job_id: str,
    repo_full_name: str,
    docs: list[dict],                   # [{"doc_type": "api", "content": "..."}]
    custom_branch_name: str | None = None,
) -> dict:
    """Runs the PR agent. Returns PRResponse with pr_url, branch_name, files_committed."""
    payload = {
        "github_token":      github_token,
        "job_id":            job_id,
        "repo_full_name":    repo_full_name,
        "docs":              docs,
        "custom_branch_name": custom_branch_name,
    }
    response = requests.post(
        f"{BACKEND_URL}/api/github/pr",
        json=payload,
        timeout=120,
    )
    if response.status_code != 200:
        raise BackendError(_extract_error_detail(response))
    return response.json()


def _extract_error_detail(response: requests.Response) -> str:
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text
