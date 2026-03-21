"""Cost evaluator — scores token usage efficiency."""

from __future__ import annotations

from retune.core.models import EvalResult, ExecutionTrace
from retune.evaluators.base import BaseEvaluator


class CostEvaluator(BaseEvaluator):
    """Evaluates cost efficiency based on token usage.

    Scores inversely proportional to token usage relative to thresholds.
    """

    name = "cost"

    def __init__(
        self, low_tokens: int = 500, high_tokens: int = 10000, **kwargs
    ) -> None:
        self._low = low_tokens
        self._high = high_tokens

    def evaluate(self, trace: ExecutionTrace) -> EvalResult:
        total_tokens = trace.total_tokens
        total_cost = trace.total_cost

        if total_tokens <= self._low:
            score = 1.0
        elif total_tokens >= self._high:
            score = 0.0
        else:
            score = 1.0 - (total_tokens - self._low) / (self._high - self._low)

        return EvalResult(
            evaluator_name=self.name,
            score=round(score, 3),
            reasoning=f"Used {total_tokens} tokens (${total_cost:.4f}).",
            details={
                "total_tokens": total_tokens,
                "total_cost_usd": total_cost,
            },
        )
