"""Prompt analyzer tool — analyzes system prompt quality and structure."""

from __future__ import annotations

import re
from typing import Any

from retune.tools.base import RetuneTool


class PromptAnalyzerTool(RetuneTool):
    """Analyzes a system prompt for quality signals.

    Checks for: structure, specificity, constraints, examples,
    format instructions, role definition, and common weaknesses.
    Used by the APO (Automatic Prompt Optimization) subagent.
    """

    name: str = "prompt_analyzer"
    description: str = (
        "Analyze a system prompt for quality and structure. "
        "Input: prompt text. Output: structural analysis including "
        "word count, has_role, has_constraints, has_examples, "
        "has_format_instructions, specificity_score, and weaknesses."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "The system prompt to analyze"},
        },
        "required": ["prompt"],
    }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        prompt = kwargs.get("prompt", "")

        if not prompt:
            return {"error": "No prompt provided", "quality_score": 0.0}

        prompt_lower = prompt.lower()
        words = prompt.split()
        word_count = len(words)
        sentences = [s.strip() for s in re.split(r'[.!?]', prompt) if s.strip()]

        # Structural checks
        has_role = any(kw in prompt_lower for kw in [
            "you are", "your role", "act as", "you're a", "as a",
        ])
        has_constraints = any(kw in prompt_lower for kw in [
            "do not", "don't", "never", "always", "must", "avoid",
            "important:", "rule:", "constraint:",
        ])
        has_examples = any(kw in prompt_lower for kw in [
            "example:", "for example", "e.g.", "such as",
            "here is an example", "sample:",
        ])
        has_format = any(kw in prompt_lower for kw in [
            "format:", "respond in", "output format", "json",
            "markdown", "bullet", "numbered list",
        ])
        has_steps = any(kw in prompt_lower for kw in [
            "step 1", "step 2", "first,", "second,", "then,",
            "1.", "2.", "3.",
        ])
        has_tool_instructions = any(kw in prompt_lower for kw in [
            "tool", "function", "api", "call", "use the",
        ])

        # Specificity score (0-1)
        specificity = 0.0
        if has_role:
            specificity += 0.15
        if has_constraints:
            specificity += 0.20
        if has_examples:
            specificity += 0.20
        if has_format:
            specificity += 0.15
        if has_steps:
            specificity += 0.15
        if has_tool_instructions:
            specificity += 0.15
        if word_count >= 50:
            specificity = min(specificity, 1.0)
        elif word_count < 20:
            specificity *= 0.5  # Short prompts are less specific

        # Identify weaknesses
        weaknesses = []
        if not has_role:
            weaknesses.append("Missing role definition — agent doesn't know WHO it is")
        if not has_constraints:
            weaknesses.append("No constraints/rules — agent has no guardrails")
        if not has_examples:
            weaknesses.append("No examples — agent learns better with demonstrations")
        if not has_format:
            weaknesses.append("No output format — responses may be inconsistent")
        if not has_steps:
            weaknesses.append("No step-by-step instructions — agent may skip reasoning")
        if word_count < 20:
            weaknesses.append(f"Very short prompt ({word_count} words) — likely too vague")
        if word_count > 2000:
            weaknesses.append(f"Very long prompt ({word_count} words) — may confuse the model")

        # Strengths
        strengths = []
        if has_role:
            strengths.append("Clear role definition")
        if has_constraints:
            strengths.append("Has explicit constraints/rules")
        if has_examples:
            strengths.append("Includes examples")
        if has_steps:
            strengths.append("Step-by-step structure")
        if has_tool_instructions:
            strengths.append("Tool usage instructions")

        return {
            "word_count": word_count,
            "sentence_count": len(sentences),
            "has_role": has_role,
            "has_constraints": has_constraints,
            "has_examples": has_examples,
            "has_format_instructions": has_format,
            "has_step_by_step": has_steps,
            "has_tool_instructions": has_tool_instructions,
            "specificity_score": round(specificity, 2),
            "strengths": strengths,
            "weaknesses": weaknesses,
            "quality_score": round(specificity, 2),
        }
