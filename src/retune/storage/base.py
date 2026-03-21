"""Abstract storage interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from retune.core.models import ExecutionTrace, OptimizationConfig


class BaseStorage(ABC):
    """Abstract base for trace and config persistence."""

    @abstractmethod
    def save_trace(self, trace: ExecutionTrace) -> None:
        """Persist an execution trace."""

    @abstractmethod
    def get_trace(self, trace_id: str) -> ExecutionTrace | None:
        """Retrieve a trace by ID."""

    @abstractmethod
    def get_traces(
        self,
        limit: int = 50,
        session_id: str | None = None,
    ) -> list[ExecutionTrace]:
        """Retrieve recent traces, optionally filtered by session."""

    @abstractmethod
    def save_config(self, name: str, config: OptimizationConfig) -> None:
        """Save a named configuration."""

    @abstractmethod
    def get_config(self, name: str) -> OptimizationConfig | None:
        """Retrieve a named configuration."""

    @abstractmethod
    def list_configs(self) -> list[str]:
        """List all saved config names."""
