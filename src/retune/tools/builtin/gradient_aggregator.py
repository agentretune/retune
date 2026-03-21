"""Gradient aggregator tool — merges multiple critiques into a unified textual gradient."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from retune.tools.base import RetuneTool

logger = logging.getLogger(__name__)


class GradientAggregatorTool(RetuneTool):
    """Aggregates multiple textual gradients (critiques) into a unified gradient.

    When beam search generates critiques from multiple traces, this tool
    merges them into a coherent, deduplicated set of improvement directions.
    """

    name: str = "gradient_aggregator"
    description: str = (
        "Merge multiple textual gradients (critiques) into a single unified gradient. "
        "Input: list of critique strings. Output: unified_gradient, themes, priority_order."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "critiques": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of textual gradient strings to merge",
            },
        },
        "required": ["critiques"],
    }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        critiques = kwargs.get("critiques", [])

        if not critiques:
            return {
                "unified_gradient": "",
                "themes": [],
                "priority_order": [],
            }

        if len(critiques) == 1:
            return {
                "unified_gradient": critiques[0],
                "themes": ["single_critique"],
                "priority_order": [0],
            }

        try:
            from retune.core.llm import create_llm

            llm = create_llm(temperature=0)

            critiques_text = "\n\n".join(
                f"CRITIQUE {i+1}:\n{c[:500]}" for i, c in enumerate(critiques)
            )

            prompt = (
                "You are merging multiple critiques of an AI agent's system prompt.\n\n"
                f"{critiques_text}\n\n"
                "Synthesize these into a SINGLE unified critique that:\n"
                "1. Deduplicates overlapping points\n"
                "2. Identifies the most common/important themes\n"
                "3. Prioritizes by frequency and severity\n\n"
                "Respond in JSON:\n"
                '{"unified_gradient": "<merged critique text>", '
                '"themes": ["<theme1>", "<theme2>"], '
                '"priority_order": ["<most important>", "<next>"]}'
            )

            result = llm.invoke(prompt)
            content = result.content if hasattr(result, "content") else str(result)

            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    "unified_gradient": parsed.get("unified_gradient", ""),
                    "themes": parsed.get("themes", []),
                    "priority_order": parsed.get("priority_order", []),
                }

        except (ImportError, Exception) as e:
            logger.debug(f"LLM aggregation failed, using heuristic: {e}")

        # Heuristic fallback: concatenate and deduplicate
        return self._heuristic_aggregate(critiques)

    def _heuristic_aggregate(self, critiques: list[str]) -> dict[str, Any]:
        """Heuristic aggregation when LLM is unavailable."""
        # Simple: join all critiques with separators
        unified = "\n\n---\n\n".join(critiques)

        # Extract common themes via keyword frequency
        theme_keywords = {
            "role_definition": ["role", "who", "identity"],
            "constraints": ["constraint", "guardrail", "rule", "limit"],
            "examples": ["example", "demonstration", "sample"],
            "formatting": ["format", "structure", "output"],
            "reasoning": ["reason", "step", "think", "chain"],
            "tool_usage": ["tool", "function", "api"],
            "grounding": ["ground", "source", "hallucin", "fact"],
        }

        combined_text = " ".join(critiques).lower()
        themes = []
        for theme, keywords in theme_keywords.items():
            if any(kw in combined_text for kw in keywords):
                themes.append(theme)

        return {
            "unified_gradient": unified,
            "themes": themes,
            "priority_order": themes,  # ordered by detection
        }
