"""
Pydantic models for API requests and responses.

Keeping these separate from the internal dataclasses in services/chunker.py
is intentional: API schemas are a public contract and change for different
reasons than internal data structures do.
"""
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    repo_url: str = Field(
        ...,
        description="HTTPS URL of a public GitHub repository",
        examples=["https://github.com/psf/requests"],
    )


class IngestResponse(BaseModel):
    job_id: str
    repo_url: str
    status: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    repo_url: str
    status: str  # queued | cloning | walking | chunking | embedding | done | failed
    files_discovered: int = 0
    files_chunked: int = 0
    chunks_created: int = 0
    error: str | None = None


class FileInfo(BaseModel):
    path: str
    extension: str
    size_bytes: int
    language: str


class ChunkInfo(BaseModel):
    chunk_id: str
    file_path: str
    symbol_name: str | None
    chunk_type: str  # function | class | block | document
    start_line: int
    end_line: int
    language: str
    preview: str  # first ~200 chars, full text is in the vector store


class IngestResult(BaseModel):
    job_id: str
    repo_url: str
    status: str
    files: list[FileInfo]
    chunks: list[ChunkInfo]


# --- Doc generation ---

class DocGenerateRequest(BaseModel):
    job_id: str = Field(..., description="job_id returned by POST /api/ingest")
    doc_type: str = Field(
        default="api",
        description="What to generate: 'api' | 'readme' | 'guide'",
    )


class DocGenerateResponse(BaseModel):
    job_id: str
    doc_type: str
    content: str          # full generated Markdown
    files_summarised: int
    llm_calls_used: int


class DocJobStatus(BaseModel):
    """Returned by GET /api/docs/generate/status/{doc_job_id} for polling."""
    doc_job_id: str
    ingest_job_id: str
    doc_type: str
    status: str           # running | done | failed
    content: str | None = None
    files_summarised: int = 0
    llm_calls_used: int = 0
    error: str | None = None


class DocGenerateStarted(BaseModel):
    """Returned immediately by POST /api/docs/generate, contains the poll ID."""
    doc_job_id: str
    ingest_job_id: str
    doc_type: str
    message: str


# --- Q&A ---

class QARequest(BaseModel):
    job_id: str = Field(..., description="job_id of the ingested repo to query")
    question: str = Field(..., description="Natural-language question about the codebase")
    history: list[dict] = Field(
        default_factory=list,
        description="Prior turns as [{role: user|assistant, content: str}]",
    )


class QAResponse(BaseModel):
    job_id: str
    question: str
    answer: str
    sources: list[dict]   # chunk metadata for the retrieved context


# --- GitHub OAuth + PR Agent ---

class OAuthStatus(BaseModel):
    connected: bool
    github_login: str = ""
    github_name: str = ""
    avatar_url: str = ""


class DocEntry(BaseModel):
    """One item to commit in the PR, either a generated doc or a code change."""
    doc_type: str
    content: str
    # Used for code changes from Q&A: file_path="app/services/rag.py" commits there directly.
    file_path: str | None = None
    label: str | None = None


class PRRequest(BaseModel):
    job_id: str = Field(..., description="Ingest job_id whose generated docs to commit")
    github_token: str = Field(..., description="GitHub OAuth access token")
    repo_full_name: str = Field(..., description="owner/repo, auto-detected from ingest URL if not provided")
    docs: list[DocEntry] = Field(..., description="One or more docs to commit in this PR")
    custom_branch_name: str | None = Field(
        default=None,
        description="Optional branch name override. Auto-generated if omitted.",
    )


class PRResponse(BaseModel):
    pr_url: str
    branch_name: str
    pr_title: str
    pr_number: int
    forked: bool = False
    fork_name: str | None = None
    files_committed: list[str] = []   # e.g. ["docs/API_Reference.md", "docs/README.md"]
