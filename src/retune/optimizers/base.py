"""Base optimizer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from retune.core.models import ExecutionTrace, OptimizationConfig, Suggestion


class BaseOptimizer(ABC):
    """Abstract base for optimization engines.

    An optimizer analyzes traces + eval results and suggests config changes.
    """

    name: str = "base"

    @abstractmethod
    def suggest(
        self,
        traces: list[ExecutionTrace],
        current_config: OptimizationConfig,
        adapter: Any | None = None,
        validation_queries: list[str] | None = None,
    ) -> list[Suggestion]:
        """Analyze traces and suggest improvements.

        Args:
            traces: Recent execution traces with eval results
            current_config: Current optimization config
            adapter: Optional adapter for rollout verification
            validation_queries: Optional queries for rollout testing

        Returns:
            List of suggested parameter changes
        """
