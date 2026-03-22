"""Config search tool — explores the parameter space for optimization."""

from __future__ import annotations

from typing import Any, cast

from retune.tools.base import RetuneTool

# Known parameter ranges and their optimization heuristics
PARAM_SPACE = {
    "top_k": {
        "type": "int",
        "range": [1, 20],
        "default": 4,
        "description": "Number of documents to retrieve",
        "increase_reason": "More documents provide better context coverage",
        "decrease_reason": "Fewer documents reduce noise and latency",
    },
    "temperature": {
        "type": "float",
        "range": [0.0, 1.0],
        "default": 0.3,
        "description": "LLM sampling temperature",
        "increase_reason": "Higher creativity for open-ended tasks",
        "decrease_reason": "More deterministic for factual/precise tasks",
    },
    "max_tokens": {
        "type": "int",
        "range": [256, 8192],
        "default": 2048,
        "description": "Maximum output tokens",
        "increase_reason": "Allow longer, more detailed responses",
        "decrease_reason": "Reduce cost and latency",
    },
    "chunk_size": {
        "type": "int",
        "range": [100, 2000],
        "default": 512,
        "description": "Document chunk size for RAG",
        "increase_reason": "Larger chunks preserve more context",
        "decrease_reason": "Smaller chunks improve retrieval precision",
    },
    "chunk_overlap": {
        "type": "int",
        "range": [0, 500],
        "default": 50,
        "description": "Overlap between document chunks",
        "increase_reason": "More overlap prevents information loss at boundaries",
        "decrease_reason": "Less overlap reduces redundancy and storage",
    },
}


class ConfigSearchTool(RetuneTool):
    """Searches the parameter space for optimization candidates.

    Given a parameter name, current value, and desired direction,
    returns a list of candidate values with rationale.
    """

    name: str = "config_search"
    description: str = (
        "Search for better parameter values. "
        "Input: param_name, current_value, direction ('increase'|'decrease'|'explore'). "
        "Output: list of candidate values with rationale and confidence."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "param_name": {"type": "string", "description": "Name of the parameter to tune"},
            "current_value": {"description": "Current value of the parameter"},
            "direction": {
                "type": "string",
                "enum": ["increase", "decrease", "explore"],
                "description": "Direction to search",
            },
        },
        "required": ["param_name", "current_value", "direction"],
    }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        param_name = kwargs.get("param_name", "")
        current_value = kwargs.get("current_value")
        direction = kwargs.get("direction", "explore")

        spec = PARAM_SPACE.get(param_name)

        if not spec:
            return {
                "param_name": param_name,
                "candidates": [],
                "note": f"Unknown parameter '{param_name}'. Known: {list(PARAM_SPACE.keys())}",
            }

        candidates = []
        range_val = cast(list[float], spec["range"])
        lo: float = range_val[0]
        hi: float = range_val[1]

        if current_value is None:
            current_value = spec["default"]

        current_value = float(current_value) if spec["type"] == "float" else int(current_value)  # type: ignore[arg-type]

        if direction == "increase":
            steps = [1.25, 1.5, 2.0] if spec["type"] == "int" else [0.1, 0.2, 0.3]
            for step in steps:
                if spec["type"] == "int":
                    new_val = min(int(current_value * step), hi)
                else:
                    new_val = round(min(current_value + step, hi), 2)
                if new_val != current_value:
                    candidates.append({
                        "value": new_val,
                        "rationale": spec["increase_reason"],
                        "confidence": 0.7 if step == steps[0] else 0.5,
                    })

        elif direction == "decrease":
            steps = [0.75, 0.5, 0.25] if spec["type"] == "int" else [0.1, 0.2, 0.3]
            for step in steps:
                if spec["type"] == "int":
                    new_val = max(int(current_value * step), lo)
                else:
                    new_val = round(max(current_value - step, lo), 2)
                if new_val != current_value:
                    candidates.append({
                        "value": new_val,
                        "rationale": spec["decrease_reason"],
                        "confidence": 0.7 if step == steps[0] else 0.5,
                    })

        elif direction == "explore":
            # Generate a spread of values around the default
            default_val = cast(float, spec["default"])
            explore_vals: list[float] = [lo, default_val, hi]
            midpoints = [(lo + default_val) / 2, (default_val + hi) / 2]
            explore_vals.extend(midpoints)
            explore_vals = sorted(set(explore_vals))

            for val in explore_vals:
                if spec["type"] == "int":
                    val = int(val)
                else:
                    val = round(val, 2)
                if val != current_value:
                    candidates.append({
                        "value": val,
                        "rationale": f"Exploring parameter space for {param_name}",
                        "confidence": 0.4,
                    })

        # Deduplicate
        seen = set()
        unique = []
        for c in candidates:
            if c["value"] not in seen:
                seen.add(c["value"])
                unique.append(c)

        return {
            "param_name": param_name,
            "current_value": current_value,
            "direction": direction,
            "candidates": unique[:5],
            "param_description": spec["description"],
            "valid_range": spec["range"],
        }
