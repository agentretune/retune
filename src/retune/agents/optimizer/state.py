"""State definition for the Optimizer Deep Agent."""

from __future__ import annotations

from typing import Any, TypedDict


class OptimizerState(TypedDict, total=False):
    """Shared state for the optimizer LangGraph agent.

    The planner decides strategies, subagents generate suggestions,
    and the aggregator produces the final list.
    """

    # Input
    traces: list[dict[str, Any]]
    current_config: dict[str, Any]

    # Planner
    analysis_summary: dict[str, Any]
    strategies_to_run: list[str]
    strategies_completed: list[str]

    # APO (Automatic Prompt Optimization) state
    apo_evaluation: str
    apo_critique: str
    apo_rewritten_prompt: str
    apo_confidence: float

    # Beam Search APO result (from external beam search)
    beam_search_result: dict[str, Any] | None

    # ConfigTuner state
    config_suggestions: list[dict[str, Any]]

    # ToolCurator state
    tool_suggestions: list[dict[str, Any]]

    # Final output
    final_suggestions: list[dict[str, Any]]

    # LLM
    messages: list[Any]
    model: str
