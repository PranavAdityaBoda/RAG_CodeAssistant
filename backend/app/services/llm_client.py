"""Groq wrapper. Tracks RPM/daily budgets and parses retry-after on 429s."""
import re
import time
from collections import deque
from dataclasses import dataclass, field

from groq import Groq, RateLimitError, APIStatusError

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Matches "Please try again in 6.21s" or "try again in 1m30s" from Groq error bodies.
_RETRY_AFTER_RE = re.compile(
    r"try again in\s+(?:(\d+)m)?(\d+(?:\.\d+)?)s", re.IGNORECASE
)


def _parse_retry_after(error_message: str) -> float:
    """Parses Groq's suggested wait time from a 429 body. Falls back to 15s."""
    match = _RETRY_AFTER_RE.search(error_message)
    if match:
        minutes = int(match.group(1) or 0)
        seconds = float(match.group(2))
        return minutes * 60 + seconds + 1.0  # 1s buffer
    return 15.0  # conservative default


@dataclass
class _RateLimitTracker:
    """Sliding window counters for RPM and daily budgets."""
    rpm_limit: int
    daily_limit: int
    _minute_window: deque = field(default_factory=deque)
    _day_count: int = 0
    _day_start: float = field(default_factory=time.time)

    def _reset_day_if_needed(self) -> None:
        if time.time() - self._day_start >= 86_400:
            self._day_count = 0
            self._day_start = time.time()

    def _prune_minute_window(self) -> None:
        cutoff = time.time() - 60
        while self._minute_window and self._minute_window[0] < cutoff:
            self._minute_window.popleft()

    def wait_if_needed(self) -> None:
        """Block until it's safe to make another request."""
        self._reset_day_if_needed()

        if self._day_count >= self.daily_limit:
            raise RuntimeError(
                f"Groq free-tier daily request cap ({self.daily_limit}) reached. "
                "Wait until midnight UTC or upgrade your plan."
            )

        self._prune_minute_window()
        if len(self._minute_window) >= self.rpm_limit:
            oldest = self._minute_window[0]
            sleep_secs = 60 - (time.time() - oldest) + 0.5
            if sleep_secs > 0:
                logger.info(
                    "RPM cap (%d) reached, sleeping %.1fs before next LLM call",
                    self.rpm_limit, sleep_secs,
                )
                time.sleep(sleep_secs)
            self._prune_minute_window()

    def record_call(self) -> None:
        self._minute_window.append(time.time())
        self._day_count += 1

    @property
    def calls_today(self) -> int:
        self._reset_day_if_needed()
        return self._day_count

    @property
    def calls_this_minute(self) -> int:
        self._prune_minute_window()
        return len(self._minute_window)


class LLMClient:
    """
    Wraps Groq so the rest of the app never imports groq directly.

    Usage:
        from app.services.llm_client import get_llm_client, ModelTier
        client = get_llm_client()
        response = client.complete(ModelTier.FAST, system_prompt, user_prompt)
    """

    def __init__(self) -> None:
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to backend/.env and restart."
            )
        self._client = Groq(api_key=settings.groq_api_key)
        self._tracker = _RateLimitTracker(
            rpm_limit=settings.groq_requests_per_minute,
            daily_limit=settings.groq_requests_per_day,
        )

    def complete(
        self,
        model_tier: str,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 5,
        temperature: float = 0.3,
    ) -> str:
        """
        Sends a chat completion request to Groq and returns the response text.

        On 429 (rate limit), parses Groq's suggested retry delay from the
        error message and sleeps exactly that long. This avoids the blind
        exponential backoff that caused the original frontend timeout issue.
        """
        model = (
            settings.groq_fast_model
            if model_tier == ModelTier.FAST
            else settings.groq_reasoning_model
        )

        for attempt in range(1, max_retries + 1):
            self._tracker.wait_if_needed()
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                )
                self._tracker.record_call()
                return response.choices[0].message.content or ""

            except RateLimitError as exc:
                # Parse the exact wait time Groq tells us to use.
                # Much better than guessing with exponential backoff.
                error_text = str(exc)
                wait = _parse_retry_after(error_text)
                logger.warning(
                    "Groq 429 on attempt %d/%d (model=%s): %s, waiting %.1fs",
                    attempt, max_retries, model, error_text[:120], wait,
                )
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Groq rate limit exceeded after {max_retries} retries. "
                        f"Last error: {error_text[:200]}"
                    ) from exc
                time.sleep(wait)

            except APIStatusError as exc:
                error_text = str(exc)
                # 413 = payload too large. Retrying the same request will
                # never succeed, raise immediately so the caller can truncate.
                if getattr(exc, "status_code", None) == 413 or "413" in error_text:
                    raise
                logger.error("Groq API error (attempt %d): %s", attempt, exc)
                if attempt == max_retries:
                    raise
                time.sleep(5)

        return ""  # unreachable but satisfies type checkers

    @property
    def usage(self) -> dict:
        return {
            "calls_today": self._tracker.calls_today,
            "calls_this_minute": self._tracker.calls_this_minute,
            "daily_limit": settings.groq_requests_per_day,
            "rpm_limit": settings.groq_requests_per_minute,
        }


class ModelTier:
    """Constants for the two Groq model tiers."""
    FAST = "fast"             # Llama 3.1 8B, bulk/high-volume tasks
    REASONING = "reasoning"   # Llama 3.3 70B, user-facing, quality-sensitive tasks


_client_instance: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Returns the shared LLMClient instance, initialising it on first call."""
    global _client_instance
    if _client_instance is None:
        _client_instance = LLMClient()
    return _client_instance
