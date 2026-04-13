"""Run a set of evaluators over a trace dict, return {name: score} dict.

Used by the candidate runner: after executing the user's agent with
overridden config, we run the registered evaluators against the produced
trace to get real eval_scores to send back to the cloud optimizer.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class _EvaluatorLike(Protocol):
    name: str
    def evaluate(self, trace: Any) -> Any: ...


def _build_trace_object(trace_dict: dict[str, Any]) -> Any:
    """Convert a plain dict trace into whatever shape evaluators expect.

    The existing evaluators accept an `ExecutionTrace` — construct one if
    possible, else pass the dict through (duck-typed evaluators handle it).
    """
    try:
        from retune.core.models import ExecutionTrace
        # Minimum viable — fill missing fields with sensible defaults
        return ExecutionTrace(
            query=trace_dict.get("query", ""),
            response=trace_dict.get("response", ""),
            steps=trace_dict.get("steps", []),
            eval_results=trace_dict.get("eval_results", []),
            config_snapshot=trace_dict.get("config_snapshot", {}),
        )
    except Exception:
        return trace_dict


def run_evaluators_on_trace(
    evaluators: list[_EvaluatorLike],
    trace: dict[str, Any],
) -> dict[str, float]:
    """Run each evaluator, return {name: score}. Failures are logged + skipped."""
    trace_obj = _build_trace_object(trace)
    scores: dict[str, float] = {}
    for ev in evaluators:
        try:
            result = ev.evaluate(trace_obj)
            scores[ev.name] = float(result.score)
        except Exception as e:
            logger.warning("Evaluator %r failed: %s", getattr(ev, "name", ev), e)
    return scores
