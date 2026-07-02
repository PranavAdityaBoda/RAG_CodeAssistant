"""
GitHub OAuth flow. Three stateless functions called by api/github.py.
"""
import secrets

import httpx

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
_TOKEN_URL     = "https://github.com/login/oauth/access_token"
_USER_URL      = "https://api.github.com/user"

_pending_states: set[str] = set()


class OAuthError(Exception):
    pass


def build_auth_url() -> str:
    """Builds the GitHub OAuth URL. State token stored for CSRF validation on callback."""
    if not settings.github_client_id:
        raise OAuthError("GITHUB_CLIENT_ID not set.")

    state = secrets.token_hex(16)
    _pending_states.add(state)

    params = {
        "client_id":    settings.github_client_id,
        "redirect_uri": settings.github_redirect_uri,
        "scope":        "repo",
        "state":        state,
    }
    url = _AUTHORIZE_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())
    logger.info("Built OAuth URL (state=%s...)", state[:8])
    return url


def exchange_code(code: str, state: str) -> str:
    """Exchanges an auth code for an access token. Validates state first."""
    if state not in _pending_states:
        raise OAuthError("Invalid or expired OAuth state. Start the login flow again.")
    _pending_states.discard(state)

    if not settings.github_client_secret:
        raise OAuthError("GITHUB_CLIENT_SECRET not set.")

    resp = httpx.post(
        _TOKEN_URL,
        headers={"Accept": "application/json"},
        data={
            "client_id":     settings.github_client_id,
            "client_secret": settings.github_client_secret,
            "code":          code,
            "redirect_uri":  settings.github_redirect_uri,
        },
        timeout=15,
    )

    if resp.status_code != 200:
        raise OAuthError(f"Token endpoint returned {resp.status_code}: {resp.text}")

    payload = resp.json()
    if "error" in payload:
        raise OAuthError(f"GitHub error: {payload['error']}, {payload.get('error_description', '')}")

    token = payload.get("access_token", "")
    if not token:
        raise OAuthError("GitHub returned an empty token.")

    logger.info("Token exchange successful")
    return token


def get_github_user(token: str) -> dict:
    """Returns the authenticated user's login, name and avatar_url."""
    resp = httpx.get(
        _USER_URL,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        timeout=10,
    )

    if resp.status_code == 401:
        raise OAuthError("Token is invalid or expired.")
    if resp.status_code != 200:
        raise OAuthError(f"User API returned {resp.status_code}: {resp.text}")

    d = resp.json()
    return {
        "login":      d.get("login", ""),
        "name":       d.get("name") or d.get("login", ""),
        "avatar_url": d.get("avatar_url", ""),
    }
