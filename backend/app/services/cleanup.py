"""
Keeps disk usage bounded on long-running deploys.

Chroma's delete_collection() doesn't actually shrink the sqlite file :
hence the VACUUM after each batch. Found this out the hard way.
"""
import shutil
import sqlite3

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _chroma():
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _drop_collections(job_ids: list[str]) -> int:
    if not job_ids:
        return 0
    try:
        client = _chroma()
        live = {c.name for c in client.list_collections()}
        dropped = 0
        for jid in job_ids:
            name = f"code_chunks_{jid}"
            if name in live:
                client.delete_collection(name)
                dropped += 1
                logger.info("Dropped collection %s", name)

        if dropped:
            db = settings.chroma_dir / "chroma.sqlite3"
            if db.exists():
                conn = sqlite3.connect(db)
                conn.execute("VACUUM")
                conn.close()
        return dropped
    except Exception as exc:
        logger.warning("Chroma cleanup failed (non-fatal): %s", exc)
        return 0


def purge_old_jobs() -> int:
    if not settings.sqlite_path.exists():
        return 0

    conn = sqlite3.connect(settings.sqlite_path)
    try:
        rows = conn.execute(
            "SELECT job_id, repo_url FROM jobs WHERE status IN ('done','failed') ORDER BY rowid DESC"
        ).fetchall()
        if not rows:
            return 0

        keep: set[str] = set()
        per_url: dict[str, int] = {}
        for jid, url in rows:
            if len(keep) >= settings.max_stored_jobs:
                break
            if per_url.get(url, 0) < 3:
                keep.add(jid)
                per_url[url] = per_url.get(url, 0) + 1

        to_drop = [r[0] for r in rows if r[0] not in keep]
        if not to_drop:
            return 0

        _drop_collections(to_drop)
        placeholders = ",".join("?" * len(to_drop))
        with conn:
            conn.execute(f"DELETE FROM jobs WHERE job_id IN ({placeholders})", to_drop)
        logger.info("Purged %d jobs (kept %d)", len(to_drop), len(keep))
        return len(to_drop)
    finally:
        conn.close()


def purge_previous_jobs_for_url(repo_url: str) -> int:
    """Wipe old jobs for this URL before a fresh ingest so we don't stack up collections."""
    if not settings.sqlite_path.exists():
        return 0

    conn = sqlite3.connect(settings.sqlite_path)
    try:
        old = conn.execute(
            "SELECT job_id FROM jobs WHERE repo_url = ? AND status IN ('done','failed')",
            (repo_url,),
        ).fetchall()
        if not old:
            return 0

        ids = [r[0] for r in old]
        _drop_collections(ids)
        placeholders = ",".join("?" * len(ids))
        with conn:
            conn.execute(f"DELETE FROM jobs WHERE job_id IN ({placeholders})", ids)
        logger.info("Cleared %d old job(s) for %s", len(ids), repo_url)
        return len(ids)
    finally:
        conn.close()


def purge_stale_clones() -> int:
    """Handles clones left over from crashed ingestion runs."""
    if not settings.clone_dir.exists():
        return 0
    n = 0
    for entry in settings.clone_dir.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
            n += 1
    return n
