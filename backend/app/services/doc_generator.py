"""
Two-stage doc generation: summarise each file with the fast model,
then assemble the final doc with the reasoning model.
"""
import time
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger
from app.services import vector_store
from app.services.llm_client import get_llm_client
from app.services.router import route_task

logger = get_logger(__name__)

_SKIP_FILENAMES: frozenset[str] = frozenset({
    "__init__.py", "__version__.py", "__main__.py",
    "conftest.py", "setup.cfg", "setup.py", "pyproject.toml",
    "requirements.txt", "requirements-dev.txt", "Makefile",
    ".env", ".env.example", ".gitignore", "LICENSE", "CHANGELOG.md",
})

_SKIP_PATH_FRAGMENTS: tuple[str, ...] = (
    "/tests/", "tests/", "/migrations/", "migrations/",
    "/fixtures/", "fixtures/", "/mock", "/stub",
)


def _is_trivial(file_path: str) -> bool:
    """Returns True for files that would produce useless documentation."""
    name = Path(file_path).name
    if name in _SKIP_FILENAMES:
        return True
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    lower = file_path.lower()
    return any(frag in lower for frag in _SKIP_PATH_FRAGMENTS)


_SUMMARISE_SYSTEM = """\
You are a software engineer writing terse internal docs.
Summarise the provided code in 1-2 sentences maximum.
Name the key functions/classes only. No preamble, no padding.
"""

_SUMMARISE_MULTI_SYSTEM = """\
You are a software engineer writing terse internal docs.
For each file below, write exactly one sentence describing what it does.
Format: "filename.py: one sentence.", no preamble, no padding.
"""

_SUMMARISE_USER = """\
File: {file_path}
{chunks_text}
"""

_SUMMARISE_MULTI_USER = """\
{files_block}
"""

_ASSEMBLE_SYSTEM = """\
You are a technical writer. Write concise, structured Markdown documentation.
Rules:
- Start with # Title, then ## sections, then ### subsections
- Use `backticks` for all function names, class names, and file paths
- Use **bold** for key terms
- One short paragraph or 3-5 bullets per section, no padding
- Include a small table where it aids scanning (e.g. function list, file list)
- Do NOT use --- horizontal rules anywhere
- Output only Markdown, no preamble
"""

_ASSEMBLE_USER_TEMPLATES = {
    "api": """\
Repo: {repo_identifier}
File summaries:
{summaries_text}

Write a concise API Reference with this structure:
# {repo_identifier} API Reference
## Modules
### `module.py`
(One sentence purpose.) Key functions/classes as bullets: `name(params)`, what it does.
(Repeat per module)
## Quick Reference
| Symbol | Description |
|--------|-------------|
| `name` | one line |
Keep the whole document under 500 words.
""",
    "readme": """\
Repo: {repo_identifier}
File summaries:
{summaries_text}

Write a concise README with this structure:
# {repo_identifier}
## Overview
(2 sentences about what this project does.)
## Key Modules
| File | Purpose |
|------|---------|
| `file.py` | what it does |
## Usage
Short code example inferred from the main module.
Keep under 300 words.
""",
    "guide": """\
Repo: {repo_identifier}
File summaries:
{summaries_text}

Write a concise beginner guide with this structure:
# {repo_identifier}: Developer Guide
## What It Does
(2 sentences in plain English.)
## How It Works
3-5 bullet points tracing the main flow.
## Key Files
| File | Read this when... |
|------|------------------|
| `file.py` | reason |
## Start Here
(One paragraph on where to begin reading the code.)
Keep under 350 words.
""",
}


# ── helpers ───────────────────────────────────────────────────────────────

def _get_chunks_by_file(job_id: str) -> dict[str, list[dict]]:
    """Retrieves all chunks for a job grouped by file path."""
    collection = vector_store.get_collection(job_id)
    if collection.count() == 0:
        return {}
    results = collection.get(include=["documents", "metadatas"])
    by_file: dict[str, list[dict]] = {}
    for doc, meta in zip(results["documents"], results["metadatas"]):
        fp = meta.get("file_path", "unknown")
        by_file.setdefault(fp, []).append({"text": doc, "metadata": meta})
    return by_file


def _chunks_text(chunks: list[dict]) -> str:
    """Formats chunks into a compact text block for the prompt."""
    return "\n---\n".join(
        f"{c['metadata'].get('symbol_name') or c['metadata'].get('chunk_type', '')}: "
        f"{c['text'][:400]}"
        for c in chunks
    )


def _summarise_batch(
    files: list[tuple[str, list[dict]]],
    client,
) -> dict[str, str]:
    """
    Summarises multiple files in a single LLM call.

    Used for files with only 1-2 chunks (small helpers, thin wrappers)
    where one call per file would waste most of the token budget on
    prompt overhead rather than actual content.

    Returns a dict of {file_path: summary}.
    """
    files_block = "\n\n".join(
        f"=== {fp} ===\n{_chunks_text(chunks)}"
        for fp, chunks in files
    )
    tier = route_task("chunk_summarise")
    raw = client.complete(
        model_tier=tier,
        system_prompt=_SUMMARISE_MULTI_SYSTEM,
        user_prompt=_SUMMARISE_MULTI_USER.format(files_block=files_block),
    )
    time.sleep(2)

    # Parse "filename.py: sentence." lines from the response.
    summaries: dict[str, str] = {}
    for fp, _ in files:
        name = Path(fp).name
        for line in raw.splitlines():
            if name in line:
                # Strip the "filename.py: " prefix if present
                parts = line.split(":", 1)
                summaries[fp] = parts[-1].strip() if len(parts) > 1 else line.strip()
                break
        else:
            # Fallback: couldn't parse this file's line, use a slice of the raw response
            summaries[fp] = raw.strip()[:120]

    return summaries


def _summarise_file_solo(
    file_path: str,
    chunks: list[dict],
    client,
    batch_size: int,
) -> str:
    """
    Summarises one file that's large enough to need its own call(s).
    Used when a file has more than 2 chunks.
    """
    language = chunks[0]["metadata"].get("language", "unknown") if chunks else "unknown"
    batch_summaries: list[str] = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        tier = route_task("chunk_summarise")
        summary = client.complete(
            model_tier=tier,
            system_prompt=_SUMMARISE_SYSTEM,
            user_prompt=_SUMMARISE_USER.format(
                file_path=file_path,
                chunks_text=_chunks_text(batch),
            ),
        )
        batch_summaries.append(summary.strip())
        time.sleep(2)

    return " ".join(batch_summaries)  # join as one paragraph, not separate blocks


# ── public API ────────────────────────────────────────────────────────────

def generate_docs(
    job_id: str,
    repo_url: str,
    doc_type: str = "api",
) -> dict:
    """
    Runs the two-stage doc generation pipeline.

    Returns:
      content:          full generated Markdown string
      files_summarised: number of non-trivial files processed
      llm_calls_used:   total Groq calls made this session
    """
    if doc_type not in _ASSEMBLE_USER_TEMPLATES:
        raise ValueError(
            f"Unknown doc_type '{doc_type}'. "
            f"Valid options: {list(_ASSEMBLE_USER_TEMPLATES.keys())}"
        )

    client = get_llm_client()
    calls_before = client.usage["calls_today"]

    # ── retrieve + filter ─────────────────────────────────────────────────
    by_file = _get_chunks_by_file(job_id)
    if not by_file:
        return {
            "content": "No chunks found. Run ingestion first.",
            "files_summarised": 0,
            "llm_calls_used": 0,
        }

    small_files: list[tuple[str, list[dict]]] = []   # 1-2 chunks
    large_files: list[tuple[str, list[dict]]] = []   # 3+ chunks
    skipped = 0

    for fp, chunks in by_file.items():
        if _is_trivial(fp):
            skipped += 1
            continue
        if len(chunks) <= 2:
            small_files.append((fp, chunks))
        else:
            large_files.append((fp, chunks))

    logger.info(
        "Doc generation: %d large files (solo), %d small files (batched), %d skipped",
        len(large_files), len(small_files), skipped,
    )

    file_summaries: dict[str, str] = {}

    # so the prompt stays under the TPM limit.
    small_batch_size = 5
    for i in range(0, len(small_files), small_batch_size):
        batch = small_files[i:i + small_batch_size]
        logger.info(
            "Batching %d small files into one summarise call (%s...)",
            len(batch), batch[0][0],
        )
        file_summaries.update(_summarise_batch(batch, client))

    # Solo-summarise large files
    total_large = len(large_files)
    for idx, (fp, chunks) in enumerate(large_files, 1):
        logger.info("[%d/%d] Summarising %s (%d chunks)", idx, total_large, fp, len(chunks))
        file_summaries[fp] = _summarise_file_solo(
            fp, chunks, client, settings.summarise_batch_size
        )

    # Each summary is truncated to 80 chars before assembly.
    # This is the crux insight: the assembly model doesn't need the full
    # 128 files × 80 chars × ~0.75 tokens/char ≈ 7,680 tokens, safely
    # under the 12,000 TPM ceiling with room for the prompt and response.
    SUMMARY_TRUNCATE = 80
    ASSEMBLY_TOKEN_BUDGET = 7_000  # conservative ceiling for the summaries block

    truncated_summaries = {
        fp: summary[:SUMMARY_TRUNCATE].rstrip()
        for fp, summary in file_summaries.items()
    }

    def _build_summaries_text(items: list[tuple[str, str]]) -> str:
        return "\n".join(f"- {fp}: {s}" for fp, s in items)

    repo_identifier = repo_url.rstrip("/").split("/")[-1]
    tier = route_task(
        "doc_assemble" if doc_type == "api"
        else "readme_generate" if doc_type == "readme"
        else "guide_generate"
    )

    all_items = list(truncated_summaries.items())
    total_chars = sum(len(fp) + len(s) + 5 for fp, s in all_items)
    estimated_tokens = total_chars // 4

    if estimated_tokens <= ASSEMBLY_TOKEN_BUDGET:
        summaries_text = _build_summaries_text(all_items)
        assemble_prompt = _ASSEMBLE_USER_TEMPLATES[doc_type].format(
            repo_identifier=repo_identifier,
            summaries_text=summaries_text,
        )
        logger.info(
            "Assembling %s document in one call (~%d tokens, %d files)",
            doc_type, estimated_tokens, len(all_items),
        )
        final_doc = client.complete(
            model_tier=tier,
            system_prompt=_ASSEMBLE_SYSTEM,
            user_prompt=assemble_prompt,
        )
    else:
        chars_per_group = ASSEMBLY_TOKEN_BUDGET * 4
        groups: list[list[tuple[str, str]]] = []
        current: list[tuple[str, str]] = []
        current_chars = 0
        for item in all_items:
            item_chars = len(item[0]) + len(item[1]) + 5
            if current_chars + item_chars > chars_per_group and current:
                groups.append(current)
                current = []
                current_chars = 0
            current.append(item)
            current_chars += item_chars
        if current:
            groups.append(current)

        logger.info(
            "Assembly too large (~%d tokens), splitting into %d groups",
            estimated_tokens, len(groups),
        )

        partial_docs: list[str] = []
        for g_idx, group in enumerate(groups, 1):
            summaries_text = _build_summaries_text(group)
            assemble_prompt = _ASSEMBLE_USER_TEMPLATES[doc_type].format(
                repo_identifier=f"{repo_identifier} (part {g_idx}/{len(groups)})",
                summaries_text=summaries_text,
            )
            logger.info("Assembling group %d/%d (%d files)", g_idx, len(groups), len(group))
            partial = client.complete(
                model_tier=tier,
                system_prompt=_ASSEMBLE_SYSTEM,
                user_prompt=assemble_prompt,
            )
            partial_docs.append(partial.strip())
            time.sleep(2)

        # Merge partials
        if len(partial_docs) == 1:
            final_doc = partial_docs[0]
        else:
            merge_prompt = (
                f"Merge these {len(partial_docs)} partial {doc_type} documentation "
                f"sections for '{repo_identifier}' into one coherent Markdown document. "
                f"Remove duplicate headings. Keep it under 400 words total.\n\n"
                + "\n\n---\n\n".join(partial_docs)
            )
            logger.info("Merging %d partial docs", len(partial_docs))
            final_doc = client.complete(
                model_tier=tier,
                system_prompt=_ASSEMBLE_SYSTEM,
                user_prompt=merge_prompt[:8000],  # hard cap on merge input
            )

    calls_used = client.usage["calls_today"] - calls_before

    return {
        "content": final_doc.strip(),
        "files_summarised": len(file_summaries),
        "llm_calls_used": calls_used,
    }
