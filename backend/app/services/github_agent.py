"""
GitHub PR agent, fork detection, branch creation, file commits, PR open.

Handles owned repos and public forks transparently. The caller just passes
a repo slug and a list of files; the agent figures out the rest.
"""
import time

from github import Github, GithubException

from app.core.config import settings
from app.core.logging import get_logger
from app.services.llm_client import get_llm_client
from app.services.router import route_task

logger = get_logger(__name__)


class PRAgentError(Exception):
    pass


_DOC_NAMES = {
    "api":    "API_Reference",
    "readme": "README",
    "guide":  "Beginner_Guide",
}

def _doc_path(doc_type: str) -> str:
    name = _DOC_NAMES.get(doc_type, doc_type.capitalize() + "_Docs")
    return f"docs/{name}.md"

def _commit_msg(doc_type: str) -> str:
    if doc_type == "code_change":
        return "fix: apply suggested code change"
    return f"docs: add {_DOC_NAMES.get(doc_type, doc_type)}"


_PR_SYSTEM = """\
You are a developer writing a GitHub pull request description.
Write a PR title and body for a documentation PR.
Format: title on first line, blank line, then the body.
Title under 72 chars, conventional commits style (docs: ...).
Body under 100 words. Don't mention AI or tooling.
"""

_PR_USER = """\
Repo: {repo_name}
Doc type: {doc_type}
Preview:
{doc_preview}

Write the PR title and body.
"""


def _whoami(gh: Github) -> str:
    return gh.get_user().login


def _get_fork(gh: Github, upstream, me: str):
    """Returns our fork of upstream, creating one if needed."""
    if upstream.owner.login == me:
        return upstream

    try:
        candidate = gh.get_repo(f"{me}/{upstream.name}")
        if candidate.fork and candidate.parent and candidate.parent.full_name == upstream.full_name:
            logger.info("Using existing fork %s", candidate.full_name)
            return candidate
        # same name, different repo, fall through and let GitHub rename the new fork
        logger.warning("%s/%s exists but isn't a fork of %s, forking anyway", me, upstream.name, upstream.full_name)
    except GithubException as e:
        if e.status != 404:
            raise

    logger.info("Forking %s...", upstream.full_name)
    fork = upstream.create_fork()

    for _ in range(15):
        time.sleep(1)
        try:
            fork.get_branch(fork.default_branch)
            logger.info("Fork ready: %s", fork.full_name)
            return fork
        except GithubException:
            pass

    raise PRAgentError("Fork created but took too long to initialise. Try again in a minute.")


def _get_head_sha(repo, default_branch: str) -> str:
    """
    Fetch HEAD SHA for the branch. PyGithub caches aggressively so we
    hit the REST endpoint directly, stale cache was causing 'sha not supplied'
    on freshly created forks.
    """
    for attempt in range(10):
        try:
            _, data = repo._requester.requestJsonAndCheck(
                "GET", f"{repo.url}/branches/{default_branch}"
            )
            sha = data.get("commit", {}).get("sha", "")
            if sha:
                return sha
        except Exception:
            try:
                sha = repo.get_branch(default_branch).commit.sha
                if sha:
                    return sha
            except Exception:
                pass
        logger.info("Waiting for branch SHA... (%d/10)", attempt + 1)
        time.sleep(3)
    raise PRAgentError(f"Couldn't read HEAD SHA for {repo.full_name}/{default_branch} after 10 tries.")


def _make_branch(repo, branch_name: str) -> None:
    sha = _get_head_sha(repo, repo.default_branch)
    repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=sha)
    logger.info("Created branch %s @ %s", branch_name, sha[:8])


def _write_file(repo, branch: str, path: str, content: str, msg: str) -> None:
    """Create or update, create_file blows up if the path already exists."""
    try:
        existing = repo.get_contents(path, ref=branch)
        repo.update_file(path=path, message=msg, content=content, sha=existing.sha, branch=branch)
        logger.info("Updated %s", path)
    except GithubException as e:
        if e.status != 404:
            raise
        repo.create_file(path=path, message=msg, content=content, branch=branch)
        logger.info("Created %s", path)


def _write_pr_description(repo_name: str, doc_type: str, content: str) -> tuple[str, str]:
    try:
        client = get_llm_client()
        raw = client.complete(
            model_tier=route_task("pr_description"),
            system_prompt=_PR_SYSTEM,
            user_prompt=_PR_USER.format(
                repo_name=repo_name,
                doc_type=_DOC_NAMES.get(doc_type, doc_type),
                doc_preview=content[:400],
            ),
            temperature=0.3,
        )
        lines = raw.strip().splitlines()
        title = lines[0].strip() if lines else f"docs: add {doc_type}"
        body = "\n".join(l for l in lines[2:] if l.strip()) if len(lines) > 2 else \
               f"Adds {_DOC_NAMES.get(doc_type, doc_type)} docs for `{repo_name}`."
        return title, body
    except Exception as exc:
        logger.warning("LLM PR description failed (%s), using fallback", exc)
        name = _DOC_NAMES.get(doc_type, doc_type)
        return f"docs: add {name}", f"Adds {name} documentation for `{repo_name}`."


def _open_pr(fork, upstream, branch: str, title: str, body: str):
    # PR lives inside the fork, branch → fork's default branch
    target = fork if fork.full_name != upstream.full_name else upstream
    base = target.default_branch
    pr = target.create_pull(title=title, body=body, head=branch, base=base)
    logger.info("Opened PR #%d: %s", pr.number, pr.html_url)
    return pr


def create_pr(
    github_token: str,
    repo_full_name: str,
    docs: list[dict],
    job_id: str,
    custom_branch_name: str | None = None,
) -> dict:
    if not docs:
        raise PRAgentError("Nothing to commit.")

    try:
        gh = Github(github_token)
        upstream = gh.get_repo(repo_full_name)
        me = _whoami(gh)
    except GithubException as e:
        msg = e.data.get("message", str(e)) if isinstance(e.data, dict) else str(e)
        raise PRAgentError(f"Can't access '{repo_full_name}': {msg}") from e

    is_own = upstream.owner.login == me
    try:
        fork = _get_fork(gh, upstream, me)
    except GithubException as e:
        msg = e.data.get("message", str(e)) if isinstance(e.data, dict) else str(e)
        raise PRAgentError(f"Fork failed: {msg}") from e

    branch = custom_branch_name.strip() if custom_branch_name and custom_branch_name.strip() \
             else f"{settings.github_pr_branch_prefix}-{int(time.time())}"

    # auto-suffix if branch already exists
    try:
        fork.get_branch(branch)
        branch = f"{branch}-{int(time.time())}"
        logger.info("Branch already existed, using %s", branch)
    except GithubException as e:
        if e.status != 404:
            raise PRAgentError(f"Branch check failed: {e}") from e

    repo_name = repo_full_name.split("/")[-1]

    try:
        _make_branch(fork, branch)

        committed = []
        for doc in docs:
            path = doc.get("file_path") or _doc_path(doc["doc_type"])
            _write_file(fork, branch, path, doc["content"], _commit_msg(doc["doc_type"]))
            committed.append(path)

        title, body = _write_pr_description(repo_name, docs[0]["doc_type"], docs[0]["content"])

        if len(committed) > 1:
            body += "\n\n**Files added:**\n" + "\n".join(f"- `{f}`" for f in committed)

        pr = _open_pr(fork, upstream, branch, title, body)

    except GithubException as e:
        msg = e.data.get("message", str(e)) if isinstance(e.data, dict) else str(e)
        errors = e.data.get("errors", []) if isinstance(e.data, dict) else []
        detail = f"{msg} ({errors})" if errors else msg
        logger.error("GitHub error: %s [%s]", detail, e.status)
        raise PRAgentError(f"GitHub API error: {detail}") from e

    return {
        "pr_url":          pr.html_url,
        "branch_name":     branch,
        "pr_title":        pr.title,
        "pr_number":       pr.number,
        "forked":          not is_own,
        "fork_name":       fork.full_name if not is_own else None,
        "files_committed": committed,
    }
