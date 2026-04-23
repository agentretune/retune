"""Usage gate -- limits deep optimization/evaluation for free tier.

Free tier: 15 deep operations (optimize + evaluate via deep agents).
Premium tier: unlimited (same API key, upgraded plan).
No API key: unlimited locally (no cloud tracking).

The gate is checked before every deep operation. It queries the cloud
API to verify the user's plan and remaining usage. Results are cached
to avoid hitting the API on every call.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Local fallback counter (when no cloud API)
_local_deep_count = 0
_FREE_TIER_LIMIT = 15
_cache: dict[str, Any] = {}
_cache_ttl = 300  # 5 minutes


class UsageGate:
    """Controls access to deep optimization/evaluation features.

    With API key:
        - Checks cloud API for plan + remaining usage
        - Free plan: 15 deep operations
        - Pro/Team/Enterprise: unlimited
        - Caches result for 5 minutes

    Without API key:
        - Local counter, 15 free operations
        - No cloud check
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._local_count = 0
        self._plan: str | None = None
        self._limit: int = _FREE_TIER_LIMIT
        self._cloud_count: int = 0
        self._last_check = 0.0

    @property
    def is_cloud(self) -> bool:
        return self._api_key is not None

    @property
    def remaining(self) -> int | None:
        """Remaining deep operations, or None if unlimited."""
        if self._plan in ("pro", "team", "enterprise"):
            return None
        used = self._cloud_count if self.is_cloud else self._local_count
        return max(0, self._limit - used)

    @property
    def is_premium(self) -> bool:
        return self._plan in ("pro", "team", "enterprise")

    def check(self, operation: str = "optimize") -> bool:
        """Check if a deep operation is allowed.

        Args:
            operation: "optimize" or "evaluate"

        Returns:
            True if allowed, False if limit reached.
        """
        # Refresh from cloud if stale
        if self.is_cloud and (time.time() - self._last_check > _cache_ttl):
            self._refresh_from_cloud()

        # Premium: always allowed
        if self.is_premium:
            return True

        # Free tier: check limit
        used = self._cloud_count if self.is_cloud else self._local_count
        if used >= self._limit:
            logger.warning(
                f"Deep {operation} limit reached ({used}/{self._limit}). "
                "Upgrade to premium for unlimited optimizations: "
                "https://agentretune.com/pricing"
            )
            return False

        return True

    def record_usage(self, operation: str = "optimize") -> None:
        """Record that a deep operation was performed."""
        self._local_count += 1

        if self.is_cloud:
            self._cloud_count += 1
            self._report_to_cloud(operation)

        remaining = self.remaining
        if remaining is not None and remaining <= 3 and remaining > 0:
            logger.info(
                f"Deep operations remaining: {remaining}/{self._limit}. "
                "Upgrade for unlimited: https://agentretune.com/pricing"
            )

    def note_preauthorize_response(self, response: dict) -> None:
        """Record runs_remaining returned by /v1/optimize/preauthorize.

        Cloud is authoritative; this is a local cache for status display.
        """
        if "runs_remaining" in response:
            remaining = int(response["runs_remaining"])
            self._cloud_count = max(0, self._limit - remaining)
            self._last_check = time.time()

    def _refresh_from_cloud(self) -> None:
        """Check cloud API for current plan and usage."""
        if not self._api_key:
            return

        try:

            # Use a quick sync request (not the background queue)
            import json
            from urllib.request import Request, urlopen

            from retune.config import settings

            url = f"{settings.cloud_base_url}/api/v1/billing/usage"
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": "retune-sdk/0.1.0",
            }
            req = Request(url, headers=headers, method="GET")
            with urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read())
                self._plan = data.get("plan", "free")
                self._limit = data.get("trace_limit", _FREE_TIER_LIMIT)
                self._cloud_count = data.get("optimization_count", 0)
                self._last_check = time.time()
                logger.debug(
                    f"Cloud usage: plan={self._plan}, "
                    f"optimizations={self._cloud_count}"
                )
        except Exception as e:
            logger.debug(f"Cloud usage check failed: {e}")
            # Don't block on cloud failure -- use local count
            self._last_check = time.time()

    def _report_to_cloud(self, operation: str) -> None:
        """Report usage to cloud API (async, non-blocking)."""
        if not self._api_key:
            return
        try:
            import json
            from urllib.request import Request, urlopen

            from retune.config import settings

            url = f"{settings.cloud_base_url}/api/v1/ingest/suggestions"
            data = json.dumps({
                "action": f"deep_{operation}",
                "session_id": "usage_tracking",
            }).encode()
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "retune-sdk/0.1.0",
            }
            req = Request(url, data=data, headers=headers, method="POST")
            urlopen(req, timeout=3.0)
        except Exception:
            pass  # Best effort

    def get_status(self) -> dict[str, Any]:
        """Get current usage gate status."""
        return {
            "is_cloud": self.is_cloud,
            "plan": self._plan or "free",
            "is_premium": self.is_premium,
            "local_count": self._local_count,
            "cloud_count": self._cloud_count,
            "limit": self._limit,
            "remaining": self.remaining,
        }
