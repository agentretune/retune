"""Metrics calculator tool — statistical analysis of scores and values."""

from __future__ import annotations

import statistics
from typing import Any

from retune.tools.base import RetuneTool


class MetricsCalculatorTool(RetuneTool):
    """Computes statistical metrics from a list of numerical values.

    Used by evaluator and optimizer agents to analyze score distributions,
    detect trends, and identify outliers.
    """

    name: str = "metrics_calculator"
    description: str = (
        "Calculate statistical metrics from a list of numbers. "
        "Returns mean, median, stddev, min, max, percentiles, and trend direction. "
        "Input: values (list of floats), optional label."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "values": {"type": "array", "items": {"type": "number"}},
            "label": {"type": "string", "description": "Optional label for the metric"},
        },
        "required": ["values"],
    }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        values = kwargs.get("values", [])
        label = kwargs.get("label", "metric")

        if not values:
            return {"label": label, "error": "No values provided", "count": 0}

        values = [float(v) for v in values]
        n = len(values)

        result: dict[str, Any] = {
            "label": label,
            "count": n,
            "mean": round(statistics.mean(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

        if n >= 2:
            result["median"] = round(statistics.median(values), 4)
            result["stddev"] = round(statistics.stdev(values), 4)

            # Percentiles
            sorted_vals = sorted(values)
            result["p25"] = round(sorted_vals[max(0, n // 4 - 1)], 4)
            result["p75"] = round(sorted_vals[min(n - 1, 3 * n // 4)], 4)

            # Trend (compare first half vs second half)
            mid = n // 2
            first_half_mean = statistics.mean(values[:mid])
            second_half_mean = statistics.mean(values[mid:])
            delta = second_half_mean - first_half_mean
            if abs(delta) < 0.01:
                result["trend"] = "stable"
            elif delta > 0:
                result["trend"] = "improving"
            else:
                result["trend"] = "degrading"
            result["trend_delta"] = round(delta, 4)
        else:
            result["median"] = result["mean"]
            result["stddev"] = 0.0
            result["trend"] = "insufficient_data"

        return result
