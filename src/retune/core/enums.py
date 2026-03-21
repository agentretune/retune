"""Enumerations for Retune."""

from __future__ import annotations

from enum import Enum


class Mode(str, Enum):
    """Execution mode — controls how much post-processing happens after each run.

    Think of it as a fan regulator:
    - OFF: no overhead, pure passthrough
    - OBSERVE: capture traces only (cheap)
    - EVALUATE: capture + run evaluators (medium)
    - IMPROVE: capture + evaluate + optimize (heavy, user-triggered)
    """

    OFF = "off"
    OBSERVE = "observe"
    EVALUATE = "evaluate"
    IMPROVE = "improve"


class StepType(str, Enum):
    """Type of step in an execution trace."""

    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    RETRIEVAL = "retrieval"
    REASONING = "reasoning"
    CUSTOM = "custom"


class SuggestionStatus(str, Enum):
    """Status of an optimization suggestion in the review pipeline."""

    PENDING = "pending"      # Generated, awaiting user review
    ACCEPTED = "accepted"    # User accepted, applied to config
    REJECTED = "rejected"    # User rejected, not applied
    APPLIED = "applied"      # Applied and verified via rollout
    REVERTED = "reverted"    # Was applied but user rolled back
