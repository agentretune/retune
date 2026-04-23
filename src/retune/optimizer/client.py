"""OptimizerClient — HTTP wrappers for the cloud /v1/optimize/* and /v1/jobs/* APIs."""
from __future__ import annotations

import json
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class OptimizerClient:
    def __init__(self, api_key: str, base_url: str = "https://api.agentretune.com") -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    def _headers(self, content: bool = False) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": "retune-sdk/0.3.0",
        }
        if content:
            h["Content-Type"] = "application/json"
        return h

    def _post(self, path: str, body: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        req = Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=self._headers(content=True),
            method="POST",
        )
        try:
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read() or b"{}")
        except HTTPError as e:
            if e.code == 402:
                raise RuntimeError(
                    "Optimization run limit reached (402). Upgrade at https://agentretune.com/pricing"
                ) from e
            raise RuntimeError(f"Cloud {path} failed: {e.code} {e.reason}") from e
        except URLError as e:
            raise RuntimeError(f"Cloud {path} unreachable: {e.reason}") from e

    def _get(self, path: str, timeout: float = 10.0) -> dict[str, Any] | None:
        url = f"{self._base_url}{path}"
        req = Request(url, headers=self._headers(), method="GET")
        try:
            with urlopen(req, timeout=timeout) as resp:
                if resp.status == 204:
                    return None
                return json.loads(resp.read() or b"{}")
        except HTTPError as e:
            raise RuntimeError(f"Cloud {path} failed: {e.code} {e.reason}") from e

    def preauthorize(
        self,
        source: str,
        n_traces: int,
        axes: list[str],
        reward_spec: dict[str, Any] | None = None,
        rewriter_llm: str | None = None,
        traces: list[dict[str, Any]] | None = None,
        tool_metadata: list[dict[str, Any]] | None = None,
        retrieval_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = {
            "source": source,
            "n_traces": n_traces,
            "axes": axes,
            "reward_spec": reward_spec,
            "rewriter_llm": rewriter_llm,
        }
        if traces is not None:
            body["traces"] = traces
        if tool_metadata is not None:
            body["tool_metadata"] = tool_metadata
        if retrieval_config is not None:
            body["retrieval_config"] = retrieval_config
        return self._post("/api/v1/optimize/preauthorize", body)

    def commit(self, run_id: str) -> dict[str, Any]:
        return self._post(f"/api/v1/optimize/{run_id}/commit", {})

    def cancel(self, run_id: str) -> dict[str, Any]:
        return self._post(f"/api/v1/optimize/{run_id}/cancel", {})

    def get_run(self, run_id: str) -> dict[str, Any]:
        out = self._get(f"/api/v1/optimize/{run_id}")
        if out is None:
            raise RuntimeError(f"Run {run_id} not found")
        return out

    def fetch_report(self, run_id: str) -> dict[str, Any]:
        out = self._get(f"/api/v1/optimize/{run_id}/report")
        if out is None:
            raise RuntimeError(f"Report for {run_id} not ready")
        return out

    def poll_pending(self, run_id: str, timeout: float = 15.0) -> dict[str, Any] | None:
        """Long-poll. Returns message dict, or None on timeout."""
        client_timeout = timeout + 5.0
        return self._get(
            f"/api/v1/jobs/pending?run_id={run_id}&timeout={timeout}",
            timeout=client_timeout,
        )

    def submit_result(
        self,
        run_id: str,
        candidate_id: str,
        trace: dict[str, Any],
        eval_scores: dict[str, float],
    ) -> dict[str, Any]:
        return self._post("/api/v1/jobs/result", {
            "run_id": run_id,
            "candidate_id": candidate_id,
            "trace": trace,
            "eval_scores": eval_scores,
        })
