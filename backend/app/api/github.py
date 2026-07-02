"""
HTTP routes for GitHub OAuth and the PR agent.

GET  /api/github/login   , returns the OAuth authorisation URL
GET  /api/github/callback, receives code from GitHub, exchanges for token,
                            redirects to Streamlit with token as query param
POST /api/github/pr      , runs the PR agent, returns PR URL
GET  /api/github/status  , returns connected GitHub username for a token

All routes delegate to services/oauth.py and services/github_agent.py.
"""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import OAuthStatus, PRRequest, PRResponse
from app.services.github_agent import PRAgentError, create_pr
from app.services.oauth import OAuthError, build_auth_url, exchange_code, get_github_user

logger = get_logger(__name__)
router = APIRouter(prefix="/api/github", tags=["github"])


@router.get("/login")
def github_login() -> dict:
    """
    Returns the GitHub OAuth authorisation URL.
    The frontend opens this in a new tab via st.link_button().
    """
    try:
        url = build_auth_url()
    except OAuthError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"auth_url": url}


@router.get("/callback")
def github_callback(code: str, state: str):
    """
    GitHub redirects here after the user grants access.

    Exchanges the code for a token, then redirects to the Streamlit
    frontend with the token as a query parameter. Streamlit reads it
    on the next rerun and stores it in session state.
    """
    try:
        token = exchange_code(code, state)
    except OAuthError as exc:
        logger.error("OAuth callback failed: %s", exc)
        # Redirect to frontend with an error message instead of a token
        error_url = f"{settings.github_frontend_url}?github_error={str(exc)[:200]}"
        return RedirectResponse(url=error_url)

    redirect_url = f"{settings.github_frontend_url}?github_token={token}"
    return RedirectResponse(url=redirect_url)


@router.get("/status", response_model=OAuthStatus)
def github_status(request: Request) -> OAuthStatus:
    """Returns GitHub user info. Token passed via Authorization: Bearer header."""
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip()
    if not token:
        return OAuthStatus(connected=False)
    try:
        user = get_github_user(token)
        return OAuthStatus(
            connected=True,
            github_login=user["login"],
            github_name=user["name"],
            avatar_url=user["avatar_url"],
        )
    except OAuthError:
        return OAuthStatus(connected=False)


@router.post("/pr", response_model=PRResponse)
def create_pull_request(request: PRRequest) -> PRResponse:
    """
    Runs the full PR agent pipeline:
      create branch → commit all docs → draft PR description → open PR.
    """
    try:
        result = create_pr(
            github_token=request.github_token,
            repo_full_name=request.repo_full_name,
            docs=[
                {"doc_type": d.doc_type, "content": d.content, "file_path": d.file_path}
                for d in request.docs
            ],
            job_id=request.job_id,
            custom_branch_name=request.custom_branch_name,
        )
    except PRAgentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"PR creation failed: {exc}"
        ) from exc

    return PRResponse(
        pr_url=result["pr_url"],
        branch_name=result["branch_name"],
        pr_title=result["pr_title"],
        pr_number=result["pr_number"],
        forked=result["forked"],
        fork_name=result["fork_name"],
        files_committed=result["files_committed"],
    )
