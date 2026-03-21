"""State definition for the Evaluator Deep Agent."""

from __future__ import annotations

from typing import Any, TypedDict


class EvaluatorState(TypedDict, total=False):
    """Shared state for the evaluator LangGraph agent.

    The supervisor sets which subagents to run, each subagent writes
    its analysis to its slot, and the synthesizer aggregates them.
    """

    # Input
    trace: dict[str, Any]

    # Supervisor routing
    subagents_to_run: list[str]
    subagents_completed: list[str]

    # Subagent outputs
    trace_analysis: dict[str, Any]
    credit_assignment: dict[str, Any]
    tool_audit: dict[str, Any]
    hallucination_check: dict[str, Any]

    # Final output
    final_eval: dict[str, Any]

    # LLM interaction
    messages: list[Any]
    model: str
