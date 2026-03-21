"""Memory store — retains patterns from past executions for learning."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from retune.core.models import ExecutionTrace


class MemoryEntry(BaseModel):
    """A single memory entry — a learned pattern or failure case."""

    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    category: str  # "failure", "success", "pattern", "config"
    query: str
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MemoryStore:
    """In-memory store for learned patterns.

    MVP implementation using a bounded deque. Post-MVP will use
    vector DB for semantic retrieval of similar past cases.
    """

    def __init__(self, max_entries: int = 1000) -> None:
        self._entries: deque[MemoryEntry] = deque(maxlen=max_entries)
        self._best_configs: dict[str, dict[str, Any]] = {}

    def add_from_trace(self, trace: ExecutionTrace) -> MemoryEntry | None:
        """Analyze a trace and store relevant patterns."""
        if not trace.eval_results:
            return None

        avg_score = trace.weighted_score or 0.0

        if avg_score < 0.5:
            category = "failure"
            summary = f"Low score ({avg_score:.2f}) for query type. "
            reasons = [r.reasoning for r in trace.eval_results if r.reasoning]
            if reasons:
                summary += " | ".join(reasons[:3])
        elif avg_score > 0.85:
            category = "success"
            summary = f"High score ({avg_score:.2f}) — good config."
        else:
            return None  # Only store notable cases

        entry = MemoryEntry(
            category=category,
            query=trace.query,
            summary=summary,
            details={
                "scores": {r.evaluator_name: r.score for r in trace.eval_results},
                "steps": len(trace.steps),
                "duration_ms": trace.duration_ms,
            },
            score=avg_score,
            config_snapshot=trace.config_snapshot,
        )
        self._entries.append(entry)

        # Track best config
        if avg_score > self._best_configs.get("_best_score", {}).get("score", 0.0):
            self._best_configs["_best_score"] = {
                "score": avg_score,
                "config": trace.config_snapshot,
            }

        return entry

    def get_failures(self, limit: int = 20) -> list[MemoryEntry]:
        """Get recent failure entries."""
        return [e for e in self._entries if e.category == "failure"][-limit:]

    def get_successes(self, limit: int = 20) -> list[MemoryEntry]:
        """Get recent success entries."""
        return [e for e in self._entries if e.category == "success"][-limit:]

    def get_best_config(self) -> dict[str, Any] | None:
        """Get the config that produced the highest score."""
        best = self._best_configs.get("_best_score")
        return best["config"] if best else None

    def get_all(self, limit: int = 100) -> list[MemoryEntry]:
        return list(self._entries)[-limit:]

    def clear(self) -> None:
        self._entries.clear()
        self._best_configs.clear()

    @property
    def size(self) -> int:
        return len(self._entries)
