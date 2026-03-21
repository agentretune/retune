"""Subagent definitions for the Optimizer Deep Agent.

Each subagent is defined as a dict with: name, description, system_prompt, tools.
These definitions are used by `create_deep_agent` to spawn isolated subagents
for APO critique, rewriting, config tuning, and tool curation.
"""

from __future__ import annotations

from typing import Any

from retune.agents.optimizer.prompts import (
    CONFIG_TUNER_PROMPT,
    CRITIQUE_PROMPT,
    REWRITE_PROMPT,
    TOOL_CURATOR_PROMPT,
)
from retune.tools.builtin.config_search import ConfigSearchTool
from retune.tools.builtin.credit_assigner import CreditAssignerTool
from retune.tools.builtin.metrics import MetricsCalculatorTool
from retune.tools.builtin.prompt_analyzer import PromptAnalyzerTool
from retune.tools.builtin.prompt_rewriter import PromptRewriterTool
from retune.tools.builtin.trace_reader import TraceReaderTool


def get_critique_subagent_definition() -> dict[str, Any]:
    """APO critique subagent: evaluates prompt and generates textual gradient."""
    return {
        "name": "prompt-critic",
        "description": (
            "APO: Evaluates current prompt against failure traces, "
            "generates textual gradient (specific, actionable critique)"
        ),
        "system_prompt": CRITIQUE_PROMPT,
        "tools": [PromptAnalyzerTool(), TraceReaderTool(), MetricsCalculatorTool()],
    }


def get_rewrite_subagent_definition() -> dict[str, Any]:
    """APO rewrite subagent: applies textual gradient to rewrite prompt."""
    return {
        "name": "prompt-rewriter",
        "description": (
            "APO: Applies textual gradient to rewrite the system prompt, "
            "producing an improved version with confidence score"
        ),
        "system_prompt": REWRITE_PROMPT,
        "tools": [PromptAnalyzerTool(), PromptRewriterTool()],
    }


def get_config_tuner_subagent_definition() -> dict[str, Any]:
    """Config tuner subagent: tunes parameters based on credit analysis."""
    return {
        "name": "config-tuner",
        "description": (
            "Tunes system parameters (top_k, temperature, etc.) based on "
            "credit assignment analysis and evaluation scores"
        ),
        "system_prompt": CONFIG_TUNER_PROMPT,
        "tools": [ConfigSearchTool(), MetricsCalculatorTool(), CreditAssignerTool()],
    }


def get_tool_curator_subagent_definition() -> dict[str, Any]:
    """Tool curator subagent: analyzes tool usage and suggests changes."""
    return {
        "name": "tool-curator",
        "description": (
            "Analyzes tool usage patterns across traces, suggests enabling, "
            "disabling, or reconfiguring tools for better efficiency"
        ),
        "system_prompt": TOOL_CURATOR_PROMPT,
        "tools": [TraceReaderTool(), MetricsCalculatorTool()],
    }


def get_all_optimizer_subagent_definitions() -> list[dict[str, Any]]:
    """Return all optimizer subagent definitions."""
    return [
        get_critique_subagent_definition(),
        get_rewrite_subagent_definition(),
        get_config_tuner_subagent_definition(),
        get_tool_curator_subagent_definition(),
    ]
