"""Base adapter interface — all framework adapters implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from retune.core.models import ExecutionTrace, OptimizationConfig


class BaseAdapter(ABC):
    """Abstract base class for framework adapters.

    An adapter wraps a specific framework's agent/chain/pipeline and:
    1. Executes it
    2. Captures execution steps into an ExecutionTrace
    3. Exposes tunable configuration
    """

    def __init__(self, agent: Any, **kwargs: Any) -> None:
        self.agent = agent
        self._kwargs = kwargs

    @abstractmethod
    def run(
        self,
        query: str,
        config: OptimizationConfig | None = None,
        **kwargs: Any,
    ) -> ExecutionTrace:
        """Execute the wrapped agent and return a structured trace.

        Args:
            query: Input query/prompt
            config: Optional optimization config to apply before execution
            **kwargs: Additional arguments passed to the agent

        Returns:
            ExecutionTrace with all captured steps
        """

    @abstractmethod
    def get_config(self) -> OptimizationConfig:
        """Extract current tunable configuration from the wrapped agent."""

    @abstractmethod
    def apply_config(self, config: OptimizationConfig) -> None:
        """Apply an optimization config to the wrapped agent.

        Only non-None fields in the config are applied.
        """

    def get_agent(self) -> Any:
        """Return the underlying agent object."""
        return self.agent
