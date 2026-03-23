"""LLM Judge evaluator — uses an LLM to score response quality."""

from __future__ import annotations

from typing import Any

from retune.config import settings
from retune.core.exceptions import EvaluatorError
from retune.core.models import EvalResult, ExecutionTrace
from retune.core.schemas import JudgeOutput
from retune.evaluators.base import BaseEvaluator

JUDGE_PROMPT = """You are an expert evaluator for AI agent/RAG system outputs.

Evaluate the following agent execution:

**User Query:** {query}

**Agent Response:** {response}
{expected_section}
**Execution Steps:**
{steps_summary}

Rate the response on these dimensions (0.0 to 1.0):

1. **Correctness**: Is the answer factually accurate and directly addresses the query?
2. **Completeness**: Does the answer cover all aspects of the query?
3. **Relevance**: Is the response relevant to what was asked?
4. **Coherence**: Is the response well-structured and easy to understand?

Rate each dimension from 0.0 to 1.0 and provide brief reasoning."""


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

        # Build expected answer section
        expected = trace.metadata.get("expected_answer")
        expected_section = ""
        if expected:
            expected_section = f"\n**Expected Answer:** {str(expected)[:1000]}\n"

        prompt = self._prompt_template.format(
            query=trace.query,
            response=str(trace.response)[:3000],
            expected_section=expected_section,
            steps_summary=steps_summary,
        )

        try:
            structured_llm = llm.with_structured_output(JudgeOutput)
            result = structured_llm.invoke(prompt)
            scores = {
                "overall_score": result.overall_score,
                "correctness": result.correctness,
                "completeness": result.completeness,
                "relevance": result.relevance,
                "coherence": result.coherence,
                "reasoning": result.reasoning,
            }
        except Exception:
            # Fallback: text-based parsing
            result = llm.invoke(prompt)
            content = result.content if hasattr(result, "content") else str(result)
            scores = self._parse_response(content)

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
        from retune.utils.json_extract import extract_json_or_default
        result = extract_json_or_default(content, {
            "overall_score": 0.5,
            "reasoning": f"Could not parse LLM judge response: {content[:200]}",
        })
        return result
