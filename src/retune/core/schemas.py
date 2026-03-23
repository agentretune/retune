"""Structured output schemas for LLM responses.

Used with langchain's `.with_structured_output()` to get guaranteed
schema-compliant responses via tool_use/function_calling.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class JudgeOutput(BaseModel):
    """Output schema for the LLM Judge evaluator."""

    overall_score: float = Field(description="Overall quality score from 0.0 to 1.0")
    correctness: float = Field(description="Factual accuracy score from 0.0 to 1.0")
    completeness: float = Field(description="Coverage of all query aspects from 0.0 to 1.0")
    relevance: float = Field(description="Relevance to the query from 0.0 to 1.0")
    coherence: float = Field(description="Structure and clarity from 0.0 to 1.0")
    reasoning: str = Field(description="Brief explanation of the scores")


class HallucinationClaim(BaseModel):
    """A single factual claim extracted from the response."""

    claim: str = Field(description="The factual claim")
    status: str = Field(description="'grounded' or 'ungrounded'")


class HallucinationResult(BaseModel):
    """Output schema for hallucination detection."""

    claims: list[HallucinationClaim] = Field(
        default_factory=list,
        description="List of claims with grounding status",
    )
    hallucination_score: float = Field(
        description="0.0 = no hallucinations, 1.0 = fully hallucinated"
    )


class SynthesisResult(BaseModel):
    """Output schema for the evaluator synthesizer."""

    overall_score: float = Field(description="Overall quality score from 0.0 to 1.0")
    correctness: float = Field(description="Correctness score")
    completeness: float = Field(description="Completeness score")
    relevance: float = Field(description="Relevance score")
    tool_efficiency: float = Field(default=1.0, description="Tool usage efficiency")
    reasoning: str = Field(description="Brief explanation of the evaluation")


class RewriteResult(BaseModel):
    """Output schema for APO prompt rewriting."""

    rewritten_prompt: str = Field(description="The improved system prompt")
    changes_made: list[str] = Field(
        default_factory=list,
        description="List of specific changes made",
    )
    confidence: float = Field(
        description="Confidence that the rewrite is better, 0.0 to 1.0"
    )


class AdditionalSuggestion(BaseModel):
    """A single config suggestion from the LLM."""

    param_name: str = Field(description="Config parameter name to change")
    new_value: float | int | str | bool = Field(description="Suggested new value")
    reasoning: str = Field(description="Why this change would help")
    confidence: float = Field(description="Confidence in this suggestion, 0.0 to 1.0")


class AdditionalSuggestions(BaseModel):
    """Output schema for LLM config analysis."""

    additional_suggestions: list[AdditionalSuggestion] = Field(
        default_factory=list,
        description="Additional config parameter changes",
    )


class PairwiseJudgeOutput(BaseModel):
    """Output schema for the pairwise LLM judge."""

    winner: str = Field(description="'A', 'B', or 'tie'")
    reasoning: str = Field(description="Why the winner is better")
    confidence: float = Field(default=0.7, description="Confidence 0.0-1.0")
    dimension_wins: dict[str, str] = Field(
        default_factory=dict,
        description="Per-dimension winner: correctness, completeness, relevance",
    )


class PromptRewriteOutput(BaseModel):
    """Output schema for the prompt rewriter tool."""

    rewritten_prompt: str = Field(description="The rewritten system prompt")
    changes_summary: str = Field(
        default="",
        description="Summary of changes made",
    )
    confidence: float = Field(
        default=0.7,
        description="Confidence the rewrite is an improvement",
    )
