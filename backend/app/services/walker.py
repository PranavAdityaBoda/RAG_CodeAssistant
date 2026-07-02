"""
Walks a cloned repo's file tree and returns the files worth chunking.

All filtering rules (ignored directories, supported extensions, size cap,
file count cap) live in one place here, driven entirely by settings, so
"what counts as a file we process" is never duplicated elsewhere.
"""
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".md": "markdown",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
}


@dataclass(frozen=True)
class DiscoveredFile:
    absolute_path: Path
    relative_path: str
    extension: str
    size_bytes: int
    language: str


def _is_ignored_dir(dir_name: str) -> bool:
    return dir_name in settings.ignored_dir_names or dir_name.startswith(".")


def walk_repo(repo_root: Path) -> list[DiscoveredFile]:
    """
    Returns every file under repo_root that passes the filters in settings,
    capped at settings.max_files_per_repo. Larger repos are truncated
    rather than rejected, so a 500-file repo still produces a usable demo.
    """
    discovered: list[DiscoveredFile] = []

    for path in sorted(repo_root.rglob("*")):
        if len(discovered) >= settings.max_files_per_repo:
            logger.warning(
                "Hit max_files_per_repo cap (%d), truncating walk",
                settings.max_files_per_repo,
            )
            break

        if path.is_dir():
            continue

        if any(_is_ignored_dir(part) for part in path.relative_to(repo_root).parts[:-1]):
            continue

        extension = path.suffix.lower()
        if extension not in settings.supported_extensions:
            continue

        try:
            size_bytes = path.stat().st_size
        except OSError:
            continue

        if size_bytes == 0 or size_bytes > settings.max_file_size_bytes:
            continue

        discovered.append(
            DiscoveredFile(
                absolute_path=path,
                relative_path=str(path.relative_to(repo_root)),
                extension=extension,
                size_bytes=size_bytes,
                language=LANGUAGE_BY_EXTENSION.get(extension, "unknown"),
            )
        )

    logger.info("Discovered %d files under %s", len(discovered), repo_root)
    return discovered
