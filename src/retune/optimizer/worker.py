"""SDKWorker — long-poll consumer that executes candidates on the SDK side."""
from __future__ import annotations

import logging
from typing import Any, Callable

from retune.optimizer.client import OptimizerClient

logger = logging.getLogger(__name__)

CandidateRunner = Callable[[dict[str, Any], list[dict[str, Any]]], tuple[dict[str, Any], dict[str, float]]]


class SDKWorker:
    """Long-polls the cloud for RunCandidate commands; runs them locally.

    The `candidate_runner` callback is what actually executes the user's
    agent with overridden config against a query set — it's supplied by
    Retuner.optimize() and encapsulates all adapter-specific logic.
    """

    def __init__(
        self,
        client: OptimizerClient,
        run_id: str,
        candidate_runner: CandidateRunner,
        poll_timeout: float = 15.0,
    ) -> None:
        self._client = client
        self._run_id = run_id
        self._runner = candidate_runner
        self._poll_timeout = poll_timeout

    def run(self) -> str:
        """Drive the worker loop to completion. Returns report_url."""
        while True:
            msg = self._client.poll_pending(self._run_id, timeout=self._poll_timeout)
            if msg is None:
                continue  # timeout — re-poll

            mtype = msg.get("type")
            if mtype == "run_candidate":
                self._handle_candidate(msg)
            elif mtype == "job_complete":
                return msg["report_url"]
            elif mtype == "job_failed":
                raise RuntimeError(f"Optimization run failed: {msg.get('reason', 'unknown')}")
            else:
                logger.warning("SDKWorker: unknown message type %r", mtype)

    def _handle_candidate(self, msg: dict[str, Any]) -> None:
        cid = msg["candidate_id"]
        overrides = msg.get("config_overrides", {})
        queries = msg.get("query_set", [])
        try:
            trace, eval_scores = self._runner(overrides, queries)
        except Exception as e:
            logger.exception("Candidate %s failed: %s", cid, e)
            # Submit a failed result — cloud treats as 0-score candidate
            trace, eval_scores = ({"error": str(e)}, {})
        self._client.submit_result(
            run_id=self._run_id,
            candidate_id=cid,
            trace=trace,
            eval_scores=eval_scores,
        )
