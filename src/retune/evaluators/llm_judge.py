"""LLM Judge evaluator — uses an LLM to score response quality."""

from __future__ import annotations

import json
import re
from typing import Any

from retune.config import settings
from retune.core.exceptions import EvaluatorError
from retune.core.models import EvalResult, ExecutionTrace
from retune.evaluators.base import BaseEvaluator

JUDGE_PROMPT = """You are an expert evaluator for AI agent/RAG system outputs.

Evaluate the following agent execution:

**User Query:** {query}

**Agent Response:** {response}

**Execution Steps:**
{steps_summary}

Rate the response on these dimensions (0.0 to 1.0):

1. **Correctness**: Is the answer factually accurate and directly addresses the query?
2. **Completeness**: Does the answer cover all aspects of the query?
3. **Relevance**: Is the response relevant to what was asked?
4. **Coherence**: Is the response well-structured and easy to understand?

Respond in this exact JSON format:
{{
    "overall_score": <float 0.0-1.0>,
    "correctness": <float>,
    "completeness": <float>,
    "relevance": <float>,
    "coherence": <float>,
    "reasoning": "<brief explanation of your scores>"
}}

Only output the JSON, nothing else."""


class LLMJudgeEvaluator(BaseEvaluator):
    """Evaluates trace quality using an LLM as a judge.

    This is the primary evaluator — it scores correctness, completeness,
    relevance, and coherence using an LLM.

    Requires: pip install retune[llm]
    """

    name = "llm_judge"

    def __init__(
        self,
        model: str | None = None,
        prompt_template: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._model_name = model or settings.eval_llm_model
        self._prompt_template = prompt_template or JUDGE_PROMPT
        self._llm = None

    def _get_llm(self) -> Any:
        if self._llm is None:
            try:
                from retune.core.llm import create_llm

                self._llm = create_llm(model=self._model_name, temperature=0)
            except ImportError:
                raise EvaluatorError(
                    "An LLM provider is required for LLM Judge. "
                    "Install with: pip install retune[llm] "
                    "or pip install langchain-anthropic, langchain-google-genai, etc."
                )
        return self._llm

    def evaluate(self, trace: ExecutionTrace) -> EvalResult:
        llm = self._get_llm()

        # Build steps summary
        steps_summary = self._format_steps(trace)

        prompt = self._prompt_template.format(
            query=trace.query,
            response=str(trace.response)[:3000],
            steps_summary=steps_summary,
        )

        try:
            result = llm.invoke(prompt)
            content = result.content if hasattr(result, "content") else str(result)
            scores = self._parse_response(content)
        except Exception as e:
            raise EvaluatorError(f"LLM Judge evaluation failed: {e}") from e

        return EvalResult(
            evaluator_name=self.name,
            score=scores.get("overall_score", 0.0),
            reasoning=scores.get("reasoning", ""),
            details={
                "correctness": scores.get("correctness", 0.0),
                "completeness": scores.get("completeness", 0.0),
                "relevance": scores.get("relevance", 0.0),
                "coherence": scores.get("coherence", 0.0),
                "model": self._model_name,
            },
        )

    def _format_steps(self, trace: ExecutionTrace) -> str:
        if not trace.steps:
            return "No detailed steps captured."

        lines = []
        for i, step in enumerate(trace.steps, 1):
            duration = (step.ended_at - step.started_at).total_seconds() * 1000
            lines.append(
                f"Step {i}: [{step.step_type.value}] {step.name} "
                f"(took {duration:.0f}ms)"
            )
            if step.step_type.value == "retrieval":
                num_docs = step.output_data.get("num_docs", "?")
                lines.append(f"  Retrieved {num_docs} documents")
            elif step.step_type.value == "tool_call":
                lines.append(f"  Tool input: {str(step.input_data)[:200]}")
                lines.append(f"  Tool output: {str(step.output_data)[:200]}")

        return "\n".join(lines)

    def _parse_response(self, content: str) -> dict[str, Any]:
        """Parse the LLM judge's JSON response."""
        # Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object in text
        json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback
        return {
            "overall_score": 0.5,
            "reasoning": f"Could not parse LLM judge response: {content[:200]}",
        }
