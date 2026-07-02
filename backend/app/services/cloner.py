"""
Handles cloning a GitHub repo to local disk so it can be walked and chunked.

Kept deliberately small and single-purpose: this module only knows how to
get a repo onto disk and clean it up afterwards. It knows nothing about
chunking, embeddings, or the API layer.
"""
import shutil
import uuid
from pathlib import Path

import git

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class CloneError(Exception):
    """Raised when a repo can't be cloned (bad URL, private repo, network, etc.)."""


def clone_repo(repo_url: str, job_id: str | None = None) -> Path:
    """
    Shallow-clones repo_url into settings.clone_dir/<job_id> and returns the path.

    A shallow clone (depth=1) is used because we only ever read the current
    snapshot of the repo, we don't need git history for this project.
    """
    job_id = job_id or uuid.uuid4().hex[:12]
    destination = settings.clone_dir / job_id
    settings.clone_dir.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        shutil.rmtree(destination)

    logger.info("Cloning %s into %s", repo_url, destination)
    try:
        git.Repo.clone_from(
            repo_url,
            destination,
            depth=settings.clone_depth,
            single_branch=True,
        )
    except git.GitCommandError as exc:
        raise CloneError(f"Could not clone '{repo_url}': {exc}") from exc

    return destination


def delete_clone(path: Path) -> None:
    """Removes a cloned repo from disk. Safe to call even if already gone."""
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
        logger.info("Deleted cloned repo at %s", path)
