"""
FastAPI app entrypoint.

Run locally with:
    uvicorn app.main:app --reload --port 8000

Data lives at ../data (outside backend/) so WatchFiles never picks up cloned repos.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.ingestion import router as ingestion_router
from app.api.docs import router as docs_router
from app.api.github import router as github_router
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title=settings.app_name,
    description="Ingest, document, query and contribute to any GitHub repository",
    version="0.4.0-day4",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingestion_router)
app.include_router(docs_router)
app.include_router(github_router)


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.environment,
    }


@app.on_event("startup")
def on_startup() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.clone_dir.mkdir(parents=True, exist_ok=True)

    # This is the only cleanup needed on startup, job purging happens after each ingestion run, not on every restart.
    from app.services.cleanup import purge_stale_clones
    stale = purge_stale_clones()
    if stale:
        logger.info("Startup: removed %d stale clone(s) from previous crash", stale)

    logger.info("%s v%s starting in '%s' mode",
                settings.app_name, "0.4.0-day4", settings.environment)
