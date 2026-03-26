"""Few-shot example optimization -- learn from the agent's own successes.

Stores high-scoring query-response pairs and injects the most relevant
ones into the system prompt as few-shot examples. This is the "experience
replay" component of the agent RL loop.
"""

from __future__ import annotations

import logging
from typing import Any

from retune.core.models import ExecutionTrace
from retune.utils.text_similarity import text_overlap_score

logger = logging.getLogger(__name__)


class FewShotOptimizer:
    """Manages few-shot examples derived from agent traces.

    Stores successful executions and retrieves the most relevant
    ones to inject as examples in the system prompt.
    """

    def __init__(
        self,
        max_examples: int = 50,
        min_score: float = 0.8,
        num_examples_to_inject: int = 3,
    ) -> None:
        self._examples: list[dict[str, Any]] = []
        self._max_examples = max_examples
        self._min_score = min_score
        self._num_inject = num_examples_to_inject

    @property
    def example_count(self) -> int:
        return len(self._examples)

    def add_from_trace(self, trace: ExecutionTrace) -> bool:
        """Add a trace as a few-shot example if quality is high enough."""
        score = trace.weighted_score
        if score is None or score < self._min_score:
            return False

        example: dict[str, Any] = {
            "query": trace.query,
            "response": str(trace.response)[:1000],
            "score": score,
            "trace_id": trace.trace_id,
        }

        # Don't add duplicates (similar queries)
        for existing in self._examples:
            if text_overlap_score(trace.query, existing["query"]) > 0.8:
                # Replace if this one scores higher
                if score > existing["score"]:
                    self._examples.remove(existing)
                    self._examples.append(example)
                    return True
                return False

        self._examples.append(example)

        # Prune to max size, keeping highest scores
        if len(self._examples) > self._max_examples:
            self._examples.sort(key=lambda x: x["score"], reverse=True)
            self._examples = self._examples[: self._max_examples]

        logger.debug(
            "Added few-shot example: '%s' (score=%.3f, total=%d)",
            trace.query[:50],
            score,
            len(self._examples),
        )
        return True

    def get_relevant_examples(
        self, query: str, n: int | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve the most relevant examples for a query."""
        if not self._examples:
            return []

        n = n or self._num_inject

        # Score each example by relevance to the query
        scored: list[tuple[float, dict[str, Any]]] = []
        for ex in self._examples:
            relevance = text_overlap_score(query, ex["query"])
            scored.append((relevance, ex))

        # Sort by relevance, take top N
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:n]]

    def build_examples_prompt(self, query: str) -> str:
        """Build a few-shot examples section for the system prompt."""
        examples = self.get_relevant_examples(query)
        if not examples:
            return ""

        lines = ["\n\nHere are examples of good responses to similar queries:\n"]
        for i, ex in enumerate(examples, 1):
            lines.append(f"Example {i}:")
            lines.append(f"  User: {ex['query']}")
            lines.append(f"  Assistant: {ex['response'][:500]}")
            lines.append("")

        return "\n".join(lines)

    def get_all_examples(self) -> list[dict[str, Any]]:
        """Get all stored examples."""
        return list(self._examples)

    def clear(self) -> None:
        """Clear all stored examples."""
        self._examples.clear()
