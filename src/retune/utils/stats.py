"""Statistical significance testing for prompt optimization."""

from __future__ import annotations

import math
import random
from typing import Any


def welch_t_test(
    scores_a: list[float], scores_b: list[float]
) -> dict[str, Any]:
    """Welch's t-test for comparing two sets of scores.

    Tests if scores_b is significantly different from scores_a.
    Uses scipy if available, otherwise a pure Python approximation.
    """
    n_a, n_b = len(scores_a), len(scores_b)
    if n_a < 2 or n_b < 2:
        return {
            "significant": False,
            "p_value": 1.0,
            "reason": "Insufficient samples (need >= 2 per group)",
        }

    mean_a = sum(scores_a) / n_a
    mean_b = sum(scores_b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in scores_a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in scores_b) / (n_b - 1)

    se = math.sqrt(var_a / n_a + var_b / n_b) if (var_a / n_a + var_b / n_b) > 0 else 1e-10
    t_stat = (mean_b - mean_a) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / denom if denom > 0 else 1.0

    # Try scipy for exact p-value
    try:
        from scipy import stats
        p_value = float(stats.t.sf(abs(t_stat), df) * 2)  # two-tailed
        method = "scipy_welch"
    except ImportError:
        # Approximate p-value using normal distribution for large df
        p_value = _approx_t_pvalue(abs(t_stat), df)
        method = "approx_welch"

    return {
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "improvement": round(mean_b - mean_a, 4),
        "df": round(df, 1),
        "method": method,
    }


def _approx_t_pvalue(t: float, df: float) -> float:
    """Approximate two-tailed p-value for t-distribution.

    Uses normal approximation for df > 30, otherwise a rough
    approximation via the regularized incomplete beta function.
    """
    if df > 30:
        # Normal approximation
        z = t
        # Abramowitz and Stegun approximation for normal CDF
        p = 0.5 * math.erfc(z / math.sqrt(2))
        return 2 * p

    # For smaller df, use a conservative approximation
    # Based on Hill's 1970 approximation
    x = df / (df + t * t)
    # Rough beta incomplete function approximation
    if t == 0:
        return 1.0
    p = x ** (df / 2) * (1 + sum(
        math.prod((df / 2 + j) for j in range(k)) / math.factorial(k) * (1 - x) ** k
        for k in range(1, min(int(df / 2) + 5, 20))
    ))
    return float(min(1.0, max(0.0, p)))


def bootstrap_ci(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Bootstrap confidence interval for difference in means.

    Pure Python, no external dependencies.
    """
    if len(scores_a) < 2 or len(scores_b) < 2:
        return {
            "significant": False,
            "reason": "Insufficient samples",
        }

    rng = random.Random(42)  # Reproducible
    diffs = []
    for _ in range(n_bootstrap):
        sample_a = rng.choices(scores_a, k=len(scores_a))
        sample_b = rng.choices(scores_b, k=len(scores_b))
        mean_a = sum(sample_a) / len(sample_a)
        mean_b = sum(sample_b) / len(sample_b)
        diffs.append(mean_b - mean_a)

    diffs.sort()
    ci_lower = diffs[int(n_bootstrap * alpha / 2)]
    ci_upper = diffs[int(n_bootstrap * (1 - alpha / 2))]
    mean_diff = sum(diffs) / len(diffs)

    return {
        "mean_diff": round(mean_diff, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "significant": ci_lower > 0 or ci_upper < 0,
        "n_bootstrap": n_bootstrap,
        "method": "bootstrap",
    }


def is_significant_improvement(
    scores_old: list[float],
    scores_new: list[float],
    alpha: float = 0.05,
    min_samples: int = 3,
) -> dict[str, Any]:
    """Test if new scores are significantly better than old scores.

    Uses Welch's t-test when enough samples, otherwise bootstrap CI.
    """
    if len(scores_old) < min_samples or len(scores_new) < min_samples:
        return {
            "significant": False,
            "reason": f"Need >= {min_samples} samples per group",
            "n_old": len(scores_old),
            "n_new": len(scores_new),
        }

    # Primary: Welch's t-test
    t_result = welch_t_test(scores_old, scores_new)

    # Secondary: Bootstrap CI for confirmation
    boot_result = bootstrap_ci(scores_old, scores_new)

    return {
        "significant": t_result["significant"] and t_result.get("improvement", 0) > 0,
        "p_value": t_result.get("p_value"),
        "improvement": t_result.get("improvement"),
        "mean_old": t_result.get("mean_a"),
        "mean_new": t_result.get("mean_b"),
        "bootstrap_ci": (boot_result.get("ci_lower"), boot_result.get("ci_upper")),
        "bootstrap_significant": boot_result.get("significant", False),
        "method": t_result.get("method"),
    }
