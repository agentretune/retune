"""Beam Search APO configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BeamSearchConfig(BaseModel):
    """Configuration for Beam Search APO (Automatic Prompt Optimization).

    Controls the search breadth (beam_width), branching (branch_factor),
    depth (beam_rounds), and cost limits for the beam search process.
    """

    beam_width: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Number of candidate prompts to keep after each round",
    )
    branch_factor: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Number of rewrites to generate per candidate per round",
    )
    beam_rounds: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Number of beam search rounds (critique→rewrite→verify cycles)",
    )
    max_rollout_queries: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum validation queries per candidate during verification rollouts",
    )
    cost_budget_usd: float = Field(
        default=0.50,
        ge=0.0,
        description="Maximum budget for the entire beam search process (USD)",
    )
    min_improvement_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum score improvement required to accept a candidate over current",
    )
    verification_enabled: bool = Field(
        default=True,
        description="Whether to run rollout verification for candidates",
    )
