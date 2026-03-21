"""Subagent definitions for the Evaluator Deep Agent.

Each subagent is defined as a dict with: name, description, system_prompt, tools.
These definitions are used by `create_deep_agent` to spawn isolated subagents
with their own context, tools, and planning capabilities.
"""

from __future__ import annotations

from typing import Any

from retune.agents.evaluator.prompts import (
    CREDIT_ASSIGNER_PROMPT,
    HALLUCINATION_DETECTOR_PROMPT,
    TOOL_AUDITOR_PROMPT,
    TRACE_ANALYZER_PROMPT,
)
from retune.tools.builtin.credit_assigner import CreditAssignerTool
from retune.tools.builtin.metrics import MetricsCalculatorTool
from retune.tools.builtin.trace_reader import TraceReaderTool


def get_trace_analyzer_definition() -> dict[str, Any]:
    """Trace analyzer subagent: step-by-step execution analysis."""
    return {
        "name": "trace-analyzer",
        "description": (
            "Analyzes execution traces step-by-step, identifies bottlenecks, "
            "timing issues, token waste, and execution flow anomalies"
        ),
        "system_prompt": TRACE_ANALYZER_PROMPT,
        "tools": [TraceReaderTool(), MetricsCalculatorTool()],
    }


def get_credit_assigner_definition() -> dict[str, Any]:
    """Credit assigner subagent: Agent Lightning hierarchical credit assignment."""
    return {
        "name": "credit-assigner",
        "description": (
            "Agent Lightning credit assignment: determines which execution step "
            "caused success or failure using causal analysis and contribution scoring"
        ),
        "system_prompt": CREDIT_ASSIGNER_PROMPT,
        "tools": [CreditAssignerTool(), TraceReaderTool(), MetricsCalculatorTool()],
    }


def get_tool_auditor_definition() -> dict[str, Any]:
    """Tool auditor subagent: audits tool usage patterns."""
    return {
        "name": "tool-auditor",
        "description": (
            "Audits tool usage: correct tool selection, input quality, "
            "output utilization, efficiency, and wasteful calls"
        ),
        "system_prompt": TOOL_AUDITOR_PROMPT,
        "tools": [TraceReaderTool(), MetricsCalculatorTool()],
    }


def get_hallucination_detector_definition() -> dict[str, Any]:
    """Hallucination detector subagent: checks grounding via LLM."""
    return {
        "name": "hallucination-detector",
        "description": (
            "Checks if the agent's response is grounded in retrieved documents "
            "and tool outputs; detects hallucinations and ungrounded claims"
        ),
        "system_prompt": HALLUCINATION_DETECTOR_PROMPT,
        "tools": [TraceReaderTool()],
    }


def get_all_evaluator_subagent_definitions() -> list[dict[str, Any]]:
    """Return all evaluator subagent definitions."""
    return [
        get_trace_analyzer_definition(),
        get_credit_assigner_definition(),
        get_tool_auditor_definition(),
        get_hallucination_detector_definition(),
    ]
