"""Adaptive threshold computation from score distributions."""
from __future__ import annotations


def compute_adaptive_threshold(
    scores: list[float],
    percentile: float = 25.0,
    min_threshold: float = 0.3,
    max_threshold: float = 0.85,
    default: float = 0.7,
) -> float:
    """Compute a threshold from the user's own score distribution.

    Uses the given percentile of scores. With fewer than 3 data points,
    falls back to the static default.

    Args:
        scores: List of scores (0.0-1.0)
        percentile: Which percentile to use as threshold
        min_threshold: Floor for computed threshold
        max_threshold: Ceiling for computed threshold
        default: Fallback when insufficient data
    """
    if len(scores) < 3:
        return default

    sorted_scores = sorted(scores)
    idx = int(len(sorted_scores) * percentile / 100.0)
    idx = max(0, min(idx, len(sorted_scores) - 1))
    threshold = sorted_scores[idx]
    return max(min_threshold, min(max_threshold, threshold))
