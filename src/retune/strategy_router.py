"""Strategy routing -- multi-armed bandit for prompt/config variants.

Maintains multiple prompt/config variants and routes queries to the
best-performing one. Uses epsilon-greedy exploration to discover
better strategies over time.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from retune.core.models import OptimizationConfig

logger = logging.getLogger(__name__)


class StrategyVariant:
    """A single prompt/config variant with tracked performance."""

    def __init__(
        self,
        name: str,
        config: OptimizationConfig,
        system_prompt: str | None = None,
    ) -> None:
        self.name = name
        self.config = config
        self.system_prompt = system_prompt
        self.scores: list[float] = []
        self.total_uses = 0

    @property
    def mean_score(self) -> float:
        if not self.scores:
            return 0.5  # Optimistic prior
        return sum(self.scores) / len(self.scores)

    def record_score(self, score: float) -> None:
        self.scores.append(score)
        self.total_uses += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mean_score": round(self.mean_score, 4),
            "total_uses": self.total_uses,
            "num_scores": len(self.scores),
            "system_prompt_preview": (self.system_prompt or "")[:100],
        }


class StrategyRouter:
    """Routes queries to the best-performing config variant.

    Uses epsilon-greedy multi-armed bandit:
    - With probability (1-epsilon): pick the best variant
    - With probability epsilon: explore a random variant
    """

    def __init__(self, epsilon: float = 0.1) -> None:
        self._variants: list[StrategyVariant] = []
        self._epsilon = epsilon
        self._active_variant: StrategyVariant | None = None

    def add_variant(
        self,
        name: str,
        config: OptimizationConfig,
        system_prompt: str | None = None,
    ) -> None:
        """Register a new strategy variant."""
        variant = StrategyVariant(
            name=name, config=config, system_prompt=system_prompt
        )
        self._variants.append(variant)
        logger.info("Added strategy variant: '%s'", name)

    def select_variant(self, query: str | None = None) -> StrategyVariant | None:
        """Select a variant using epsilon-greedy."""
        if not self._variants:
            return None

        if random.random() < self._epsilon:
            # Explore
            variant = random.choice(self._variants)
        else:
            # Exploit -- pick best mean score
            variant = max(self._variants, key=lambda v: v.mean_score)

        self._active_variant = variant
        return variant

    def record_result(self, score: float) -> None:
        """Record the score for the last selected variant."""
        if self._active_variant:
            self._active_variant.record_score(score)

    def get_best_variant(self) -> StrategyVariant | None:
        """Get the variant with the highest mean score."""
        if not self._variants:
            return None
        return max(self._variants, key=lambda v: v.mean_score)

    def get_summary(self) -> dict[str, Any]:
        best = self.get_best_variant()
        return {
            "num_variants": len(self._variants),
            "epsilon": self._epsilon,
            "variants": [v.to_dict() for v in self._variants],
            "best": best.name if best else None,
        }

    @property
    def variant_count(self) -> int:
        return len(self._variants)
