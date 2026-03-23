"""Cloud API client for agentretune.com."""

from __future__ import annotations

import json
import logging
import queue
import threading
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.agentretune.com"


class CloudClient:
    """HTTP client for the retune cloud API.

    Sends traces and evals to agentretune.com in a background thread
    so it never blocks the user's agent execution.

    Usage:
        client = CloudClient(api_key="rt-...")
        client.send_trace(trace_data)  # non-blocking
        client.flush()  # wait for pending sends
        client.close()  # shutdown background worker
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 10.0,
        max_retries: int = 2,
        max_queue_size: int = 1000,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(
            maxsize=max_queue_size
        )
        self._worker = threading.Thread(
            target=self._process_queue, daemon=True, name="retune-cloud"
        )
        self._worker.start()
        self._closed = False

    def send_trace(self, trace_data: dict[str, Any]) -> None:
        """Queue a trace for async upload to cloud."""
        if self._closed:
            return
        try:
            self._queue.put_nowait({
                "type": "trace",
                "data": trace_data,
            })
        except queue.Full:
            logger.warning("Cloud upload queue full, dropping trace")

    def send_eval(self, trace_id: str, eval_data: list[dict[str, Any]]) -> None:
        """Queue evaluation results for async upload."""
        if self._closed:
            return
        try:
            self._queue.put_nowait({
                "type": "eval",
                "data": {"trace_id": trace_id, "eval_results": eval_data},
            })
        except queue.Full:
            logger.warning("Cloud upload queue full, dropping eval")

    def send_suggestion(self, suggestion_data: dict[str, Any]) -> None:
        """Queue a suggestion event for async upload."""
        if self._closed:
            return
        try:
            self._queue.put_nowait({
                "type": "suggestion",
                "data": suggestion_data,
            })
        except queue.Full:
            pass

    def flush(self, timeout: float = 5.0) -> None:
        """Wait for all queued items to be sent."""
        self._queue.join()

    def close(self) -> None:
        """Shutdown the background worker."""
        self._closed = True
        self._queue.put(None)  # Sentinel to stop worker
        self._worker.join(timeout=5.0)

    def _process_queue(self) -> None:
        """Background worker that sends queued items."""
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break

            try:
                self._send(item)
            except Exception as e:
                logger.debug(f"Cloud upload failed: {e}")
            finally:
                self._queue.task_done()

    def _send(self, item: dict[str, Any]) -> None:
        """Send a single item to the cloud API with retries."""
        item_type = item["type"]
        data = item["data"]

        endpoint_map = {
            "trace": "/api/v1/ingest/traces",
            "eval": "/api/v1/ingest/evals",
            "suggestion": "/api/v1/ingest/suggestions",
        }
        endpoint = endpoint_map.get(item_type, "/api/v1/ingest/traces")
        url = f"{self._base_url}{endpoint}"

        payload = json.dumps(data, default=str).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": "retune-sdk/0.1.0",
        }

        for attempt in range(self._max_retries + 1):
            try:
                req = Request(url, data=payload, headers=headers, method="POST")
                with urlopen(req, timeout=self._timeout) as resp:
                    if resp.status < 300:
                        return
                    logger.debug(
                        f"Cloud API returned {resp.status} for {item_type}"
                    )
            except URLError as e:
                if attempt < self._max_retries:
                    continue
                logger.debug(
                    f"Cloud upload failed after {self._max_retries + 1} attempts: {e}"
                )

    def check_connection(self) -> bool:
        """Test if the cloud API is reachable and the API key is valid."""
        url = f"{self._base_url}/api/v1/auth/verify"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": "retune-sdk/0.1.0",
        }
        try:
            req = Request(url, headers=headers, method="GET")
            with urlopen(req, timeout=5.0) as resp:
                return bool(resp.status == 200)
        except Exception:
            return False
