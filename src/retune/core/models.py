"""Core data models — the foundation of Retune."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from retune.core.enums import Mode, StepType, SuggestionStatus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid4())


class TokenUsage(BaseModel):
    """Token usage for a single LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Step(BaseModel):
    """A single step in an execution trace.

    Every agent action — LLM call, tool use, retrieval — is captured as a Step.
    """

    step_id: str = Field(default_factory=_uuid)
    step_type: StepType
    name: str  # e.g., "retrieve_documents", "generate_answer", "search_tool"
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: datetime = Field(default_factory=_utcnow)
    token_usage: TokenUsage | None = None
    cost_usd: float | None = None

    @property
    def duration_ms(self) -> float:
        """Duration of this step in milliseconds."""
        return (self.ended_at - self.started_at).total_seconds() * 1000


class EvalResult(BaseModel):
    """Result from a single evaluator."""

    evaluator_name: str
    score: float  # 0.0 - 1.0 normalized
    reasoning: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    evaluated_at: datetime = Field(default_factory=_utcnow)


class ExecutionTrace(BaseModel):
    """Complete trace of a single agent/RAG execution.

    This is the universal format — adapters convert framework-specific
    execution data into this structure.
    """

    trace_id: str = Field(default_factory=_uuid)
    session_id: str | None = None
    query: str
    response: Any = None
    steps: list[Step] = Field(default_factory=list)
    mode: Mode = Mode.OBSERVE
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: datetime = Field(default_factory=_utcnow)
    eval_results: list[EvalResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.ended_at - self.started_at).total_seconds() * 1000

    @property
    def total_tokens(self) -> int:
        return sum(
            s.token_usage.total_tokens for s in self.steps if s.token_usage is not None
        )

    @property
    def total_cost(self) -> float:
        return sum(s.cost_usd for s in self.steps if s.cost_usd is not None)

    @property
    def weighted_score(self) -> float | None:
        """Average score across all evaluators (simple mean)."""
        if not self.eval_results:
            return None
        return sum(r.score for r in self.eval_results) / len(self.eval_results)

    def get_score(self, evaluator_name: str) -> float | None:
        for r in self.eval_results:
            if r.evaluator_name == evaluator_name:
                return r.score
        return None


class OptimizationConfig(BaseModel):
    """Tunable parameters for a wrapped system.

    Covers both RAG and Agent knobs. Fields that are None are not being tuned.
    """

    # RAG parameters
    top_k: int | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    embedding_model: str | None = None
    use_reranker: bool | None = None
    search_type: str | None = None  # "similarity", "mmr", "hybrid"

    # Agent parameters
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_strategy: str | None = None  # "react", "structured", "cot"
    max_tool_calls: int | None = None
    enabled_tools: list[str] | None = None

    # Custom parameters for framework-specific tuning
    custom_params: dict[str, Any] = Field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, Any]:
        """Return only non-None parameters."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class Suggestion(BaseModel):
    """A single optimization suggestion from the optimizer."""

    suggestion_id: str = Field(default_factory=_uuid)
    param_name: str
    old_value: Any
    new_value: Any
    reasoning: str
    confidence: float = 0.5  # 0-1
    category: str = "general"  # "rag", "agent", "prompt", "general"
    status: SuggestionStatus = SuggestionStatus.PENDING
    created_at: datetime = Field(default_factory=_utcnow)


class ExperimentResult(BaseModel):
    """Result of running an experiment (config comparison)."""

    experiment_id: str = Field(default_factory=_uuid)
    config_a: OptimizationConfig
    config_b: OptimizationConfig
    traces_a: list[str] = Field(default_factory=list)  # trace IDs
    traces_b: list[str] = Field(default_factory=list)
    score_a: float = 0.0
    score_b: float = 0.0
    winner: str = ""  # "a", "b", or "tie"
    created_at: datetime = Field(default_factory=_utcnow)


class Span(BaseModel):
    """A span links to a Step and stores contribution scores.

    Inspired by Agent Lightning's span tracing — tracks how much each step
    contributed to the overall outcome (success or failure).
    """

    span_id: str = Field(default_factory=_uuid)
    step_id: str  # Links to Step.step_id
    step_type: StepType
    name: str
    contribution_score: float = 0.0  # -1.0 (harmful) to 1.0 (helpful)
    is_bottleneck: bool = False
    reasoning: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class BeamCandidate(BaseModel):
    """A candidate prompt in the beam search process."""

    prompt: str
    score: float = 0.0
    confidence: float = 0.0
    generation: int = 0  # Which beam round produced this
    parent_prompt: str = ""  # The prompt this was derived from
    critique: str = ""  # The textual gradient used to produce this
    changes_made: list[str] = Field(default_factory=list)
    verified: bool = False
    verification_score: float = 0.0


class BeamSearchResult(BaseModel):
    """Result of a beam search APO process."""

    best_prompt: str
    best_score: float = 0.0
    baseline_score: float = 0.0
    improvement: float = 0.0  # best_score - baseline_score
    candidates_explored: int = 0
    rounds_completed: int = 0
    total_cost_usd: float = 0.0
    beam_history: list[BeamCandidate] = Field(default_factory=list)
    verified: bool = False
    statistically_significant: bool = False
    p_value: float | None = None


class WrapperResponse(BaseModel):
    """Response from Retuner.run()."""

    output: Any  # The actual agent/RAG response
    trace: ExecutionTrace | None = None
    eval_results: list[EvalResult] = Field(default_factory=list)
    suggestions: list[Suggestion] = Field(default_factory=list)
    mode: Mode = Mode.OFF
