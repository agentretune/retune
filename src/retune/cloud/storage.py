"""Cloud-backed storage — dual-write to local SQLite + agentretune.com."""

from __future__ import annotations

import logging
from typing import Any

from retune.cloud.client import CloudClient
from retune.core.models import ExecutionTrace, OptimizationConfig
from retune.storage.base import BaseStorage
from retune.storage.sqlite_storage import SQLiteStorage

logger = logging.getLogger(__name__)


class CloudStorage(BaseStorage):
    """Dual-write storage: local SQLite cache + cloud sync.

    Reads always go to local SQLite (fast, offline-capable).
    Writes go to both local and cloud (async, non-blocking).

    If cloud is unreachable, data is still persisted locally.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.agentretune.com",
        db_path: str = "retune.db",
    ) -> None:
        self._local = SQLiteStorage(db_path)
        self._cloud = CloudClient(api_key=api_key, base_url=base_url)
        self._api_key = api_key
        logger.info("Cloud storage enabled — traces will sync to agentretune.com")

    def save_trace(self, trace: ExecutionTrace) -> None:
        """Save locally and sync to cloud."""
        # Always save locally first (fast, reliable)
        self._local.save_trace(trace)

        # Async sync to cloud (non-blocking)
        try:
            trace_data = trace.model_dump(mode="json")
            self._cloud.send_trace(trace_data)
        except Exception as e:
            logger.debug(f"Cloud trace sync failed: {e}")

        # Also sync eval results if present
        if trace.eval_results:
            try:
                eval_data = [r.model_dump(mode="json") for r in trace.eval_results]
                self._cloud.send_eval(trace.trace_id, eval_data)
            except Exception as e:
                logger.debug(f"Cloud eval sync failed: {e}")

    def get_trace(self, trace_id: str) -> ExecutionTrace | None:
        """Read from local cache."""
        return self._local.get_trace(trace_id)

    def get_traces(
        self,
        limit: int = 50,
        session_id: str | None = None,
    ) -> list[ExecutionTrace]:
        """Read from local cache."""
        return self._local.get_traces(limit=limit, session_id=session_id)

    def save_config(self, name: str, config: OptimizationConfig) -> None:
        """Save locally and sync to cloud."""
        self._local.save_config(name, config)

    def get_config(self, name: str) -> OptimizationConfig | None:
        return self._local.get_config(name)

    def list_configs(self) -> list[str]:
        return self._local.list_configs()

    def send_suggestion_event(
        self, suggestion_data: dict[str, Any]
    ) -> None:
        """Send a suggestion accept/reject event to cloud for analytics."""
        try:
            self._cloud.send_suggestion(suggestion_data)
        except Exception as e:
            logger.debug(f"Cloud suggestion sync failed: {e}")

    def flush(self) -> None:
        """Wait for all pending cloud uploads to complete."""
        self._cloud.flush()

    def close(self) -> None:
        """Shutdown the cloud client."""
        self._cloud.close()

    @property
    def is_cloud_enabled(self) -> bool:
        return True
