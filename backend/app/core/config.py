"""
configuration for backend
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "CodeLens"
    environment: str = "local" 

    # --- Storage paths ---
    # Stored at ../data relative to the backend/ working directory so
    # WatchFiles never sees cloned repos and triggers a spurious reload.
    data_dir: Path = Path("../data")
    clone_dir: Path = Path("../data/clones")
    chroma_dir: Path = Path("../data/chroma")
    sqlite_path: Path = Path("../data/app.db")

    # --- Ingestion limits ---
    max_files_per_repo: int = 300
    max_file_size_bytes: int = 300_000
    clone_depth: int = 1

    supported_extensions: tuple[str, ...] = (
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".md", ".txt", ".json", ".yaml", ".yml",
    )

    ignored_dir_names: tuple[str, ...] = (
        ".git", "node_modules", "__pycache__", ".venv", "venv",
        "dist", "build", ".next", ".pytest_cache", "vendor",
        "site-packages", ".mypy_cache", ".tox", "coverage",
    )

    # --- Chunking ---
    fallback_chunk_lines: int = 60
    fallback_chunk_overlap: int = 10

    # --- Embeddings ---
    embedding_model_name: str = "all-MiniLM-L6-v2"
    use_fake_embeddings: bool = False

    # --- LLM (Groq, free tier) ---
    groq_api_key: str = ""
    groq_fast_model: str = "llama-3.1-8b-instant"
    groq_reasoning_model: str = "llama-3.3-70b-versatile"
    groq_requests_per_minute: int = 30
    groq_requests_per_day: int = 1000
    summarise_batch_size: int = 6
    qa_top_k: int = 6

    # --- CORS ---
    # Comma-separated list of allowed origins. Override for deployment.
    # e.g. ALLOWED_ORIGINS=https://your-frontend.up.railway.app
    allowed_origins: tuple[str, ...] = (
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    )

    # --- GitHub OAuth ---
    github_client_id: str = ""
    github_client_secret: str = ""
    github_redirect_uri: str = "http://localhost:8000/api/github/callback"
    github_frontend_url: str = "http://localhost:8501"
    github_pr_branch_prefix: str = "docs/updates"

    max_stored_jobs: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Singleton, import this everywhere instead of constructing Settings() again.
settings = Settings()
