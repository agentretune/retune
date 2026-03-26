"""Auto-evaluation and optimization loop controller.

This is the "reinforcement learning for agents" engine:
- Tracks call counts and triggers evaluation after N calls
- Detects performance drift by comparing recent vs baseline scores
- Manages the optimization counter (15 free, then paid)
- Orchestrates the auto-optimization cycle
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from retune.core.models import EvalResult, ExecutionTrace

logger = logging.getLogger(__name__)


class AutoEvalController:
    """Controls automatic evaluation and optimization triggers.

    Usage:
        controller = AutoEvalController(
            eval_every_n_calls=50,
            optimize_on_drift=True,
            drift_threshold=0.1,
        )

        # Called after every Retuner.run()
        controller.on_trace(trace, eval_results)

        # Check if optimization should run
        if controller.should_optimize():
            suggestions = controller.run_optimization(...)
    """

    def __init__(
        self,
        eval_every_n_calls: int = 50,
        optimize_on_drift: bool = True,
        drift_threshold: float = 0.1,
        baseline_window: int = 100,
        recent_window: int = 20,
        max_free_optimizations: int = 15,
        optimization_callback: Callable[..., Any] | None = None,
    ) -> None:
        self._eval_every = eval_every_n_calls
        self._optimize_on_drift = optimize_on_drift
        self._drift_threshold = drift_threshold
        self._baseline_window = baseline_window
        self._recent_window = recent_window
        self._max_free = max_free_optimizations
        self._optimization_callback = optimization_callback

        # State
        self._call_count = 0
        self._optimization_count = 0
        self._is_premium = False
        self._scores: list[float] = []
        self._baseline_scores: list[float] = []
        self._baseline_locked = False
        self._last_eval_at = 0
        self._last_optimize_at = 0
        self._drift_detected = False
        self._drift_amount = 0.0

    def set_premium(self, is_premium: bool) -> None:
        """Enable premium (unlimited optimizations)."""
        self._is_premium = is_premium

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def optimization_count(self) -> int:
        return self._optimization_count

    @property
    def optimizations_remaining(self) -> int | None:
        """Remaining free optimizations, or None if premium."""
        if self._is_premium:
            return None
        return max(0, self._max_free - self._optimization_count)

    @property
    def can_optimize(self) -> bool:
        """Whether optimization is allowed (premium or under free limit)."""
        return self._is_premium or self._optimization_count < self._max_free

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected

    @property
    def drift_amount(self) -> float:
        return self._drift_amount

    def on_trace(
        self, trace: ExecutionTrace, eval_results: list[EvalResult]
    ) -> dict[str, Any]:
        """Called after every agent execution. Returns status dict.

        This is the main "tick" of the RL loop.
        """
        self._call_count += 1

        # Track scores
        if eval_results:
            avg_score = sum(r.score for r in eval_results) / len(eval_results)
            self._scores.append(avg_score)

            # Lock baseline after enough data
            if not self._baseline_locked and len(self._scores) >= self._baseline_window:
                self._baseline_scores = list(self._scores[-self._baseline_window :])
                self._baseline_locked = True
                logger.info(
                    "Baseline locked: %d scores, mean=%.3f",
                    len(self._baseline_scores),
                    sum(self._baseline_scores) / len(self._baseline_scores),
                )

        # Check for drift
        self._check_drift()

        # Build status
        status: dict[str, Any] = {
            "call_count": self._call_count,
            "should_eval": self._should_eval(),
            "drift_detected": self._drift_detected,
            "drift_amount": round(self._drift_amount, 4),
            "should_optimize": self.should_optimize(),
            "optimization_count": self._optimization_count,
            "can_optimize": self.can_optimize,
        }

        return status

    def _should_eval(self) -> bool:
        """Check if it's time for an evaluation cycle."""
        return self._call_count % self._eval_every == 0

    def _check_drift(self) -> None:
        """Detect performance drift by comparing recent vs baseline."""
        if not self._baseline_locked:
            self._drift_detected = False
            self._drift_amount = 0.0
            return

        if len(self._scores) < self._recent_window:
            return

        recent = self._scores[-self._recent_window :]
        baseline_mean = sum(self._baseline_scores) / len(self._baseline_scores)
        recent_mean = sum(recent) / len(recent)

        self._drift_amount = baseline_mean - recent_mean  # positive = degradation
        self._drift_detected = self._drift_amount > self._drift_threshold

        if self._drift_detected:
            logger.warning(
                "Performance drift detected: baseline=%.3f recent=%.3f drift=%.3f",
                baseline_mean,
                recent_mean,
                self._drift_amount,
            )

    def should_optimize(self) -> bool:
        """Check if auto-optimization should trigger."""
        if not self.can_optimize:
            return False

        # Trigger on drift
        if self._optimize_on_drift and self._drift_detected:
            return True

        # Trigger on scheduled eval
        if self._should_eval() and self._scores:
            recent = self._scores[-self._recent_window :]
            recent_mean = sum(recent) / len(recent)
            if recent_mean < 0.7:  # Below acceptable threshold
                return True

        return False

    def record_optimization(self) -> bool:
        """Record that an optimization was performed.

        Returns True if allowed, False if limit reached.
        """
        if not self.can_optimize:
            return False
        self._optimization_count += 1
        self._last_optimize_at = self._call_count
        logger.info(
            "Optimization #%d recorded (remaining: %s)",
            self._optimization_count,
            self.optimizations_remaining,
        )
        return True

    def update_baseline(self) -> None:
        """Reset baseline after successful optimization."""
        if len(self._scores) >= self._recent_window:
            self._baseline_scores = list(self._scores[-self._recent_window :])
            self._drift_detected = False
            self._drift_amount = 0.0
            logger.info("Baseline updated after optimization")

    def get_summary(self) -> dict[str, Any]:
        """Get full status summary."""
        baseline_mean = (
            sum(self._baseline_scores) / len(self._baseline_scores)
            if self._baseline_scores
            else None
        )
        recent = self._scores[-self._recent_window :] if self._scores else []
        recent_mean = sum(recent) / len(recent) if recent else None

        return {
            "call_count": self._call_count,
            "optimization_count": self._optimization_count,
            "optimizations_remaining": self.optimizations_remaining,
            "is_premium": self._is_premium,
            "baseline_mean": round(baseline_mean, 4) if baseline_mean else None,
            "recent_mean": round(recent_mean, 4) if recent_mean else None,
            "drift_detected": self._drift_detected,
            "drift_amount": round(self._drift_amount, 4),
            "baseline_locked": self._baseline_locked,
            "total_scores": len(self._scores),
        }
