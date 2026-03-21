"""Base evaluator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from retune.core.models import EvalResult, ExecutionTrace


class BaseEvaluator(ABC):
    """Abstract base class for trace evaluators.

    Each evaluator scores one dimension of execution quality
    (e.g., correctness, retrieval relevance, latency, cost).
    """

    name: str = "base"

    @abstractmethod
    def evaluate(self, trace: ExecutionTrace) -> EvalResult:
        """Evaluate an execution trace and return a scored result.

        Args:
            trace: The execution trace to evaluate

        Returns:
            EvalResult with score (0.0 - 1.0), reasoning, and details
        """
