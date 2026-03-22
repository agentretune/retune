"""Latency evaluator — scores execution speed."""

from __future__ import annotations

from retune.core.models import EvalResult, ExecutionTrace
from retune.evaluators.base import BaseEvaluator


class LatencyEvaluator(BaseEvaluator):
    """Evaluates execution latency.

    Uses configurable thresholds to score:
    - < fast_ms: 1.0
    - > slow_ms: 0.0
    - Linear interpolation between
    """

    name = "latency"

    def __init__(self, fast_ms: float = 1000, slow_ms: float = 10000, **kwargs) -> None:
        self._fast_ms = fast_ms
        self._slow_ms = slow_ms

    def evaluate(self, trace: ExecutionTrace) -> EvalResult:
        duration = trace.duration_ms

        if duration <= self._fast_ms:
            score = 1.0
        elif duration >= self._slow_ms:
            score = 0.0
        else:
            score = 1.0 - (duration - self._fast_ms) / (self._slow_ms - self._fast_ms)

        # Find slowest step
        slowest_step = None
        slowest_ms: float = 0
        for step in trace.steps:
            step_ms = (step.ended_at - step.started_at).total_seconds() * 1000
            if step_ms > slowest_ms:
                slowest_ms = step_ms
                slowest_step = step.name

        return EvalResult(
            evaluator_name=self.name,
            score=round(score, 3),
            reasoning=(
                f"Total: {duration:.0f}ms. "
                f"Slowest step: {slowest_step} ({slowest_ms:.0f}ms)."
            ),
            details={
                "duration_ms": duration,
                "slowest_step": slowest_step,
                "slowest_step_ms": slowest_ms,
                "num_steps": len(trace.steps),
            },
        )
