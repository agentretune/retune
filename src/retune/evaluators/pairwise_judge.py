"""Pairwise LLM Judge — compares two responses head-to-head.

Research shows LLMs are more reliable at relative comparisons than
absolute scoring. This evaluator asks "which response is better?"
and converts to a score.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from retune.config import settings
from retune.core.models import EvalResult, ExecutionTrace
from retune.core.schemas import PairwiseJudgeOutput
from retune.evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)

PAIRWISE_PROMPT = """You are an expert evaluator comparing two AI responses.

**User Query:** {query}

**Response A:**
{response_a}

**Response B:**
{response_b}

Compare these responses on:
1. **Correctness**: Which is more factually accurate?
2. **Completeness**: Which covers more aspects of the query?
3. **Relevance**: Which is more focused on answering the query?
4. **Coherence**: Which is better structured and clearer?

Pick the overall winner: 'A', 'B', or 'tie'.
For each dimension, indicate which response wins."""


class PairwiseJudgeEvaluator(BaseEvaluator):
    """Evaluates by comparing response against a reference via pairwise comparison.

    In single-trace mode (evaluate()), compares against expected_answer from
    trace.metadata. In compare mode (compare()), compares two traces directly.

    Position bias mitigation: randomly swaps A/B positions and adjusts result.
    """

    name = "pairwise_judge"

    def __init__(self, model: str | None = None, **kwargs: Any) -> None:
        self._model_name = model or settings.eval_llm_model
        self._llm = None

    def _get_llm(self) -> Any:
        if self._llm is None:
            from retune.core.llm import create_llm
            self._llm = create_llm(model=self._model_name, temperature=0)
        return self._llm

    def evaluate(self, trace: ExecutionTrace) -> EvalResult:
        """Single-trace evaluation: compare response vs expected_answer."""
        expected = trace.metadata.get("expected_answer")
        if not expected:
            return EvalResult(
                evaluator_name=self.name,
                score=0.5,
                reasoning="No expected_answer in metadata — cannot do pairwise comparison.",
                details={"mode": "skipped"},
            )

        return self._compare_responses(
            query=trace.query,
            response_current=str(trace.response),
            response_reference=str(expected),
        )

    def compare(
        self, trace_a: ExecutionTrace, trace_b: ExecutionTrace
    ) -> EvalResult:
        """Compare two traces for the same query."""
        return self._compare_responses(
            query=trace_a.query,
            response_current=str(trace_a.response),
            response_reference=str(trace_b.response),
        )

    def _compare_responses(
        self,
        query: str,
        response_current: str,
        response_reference: str,
    ) -> EvalResult:
        """Core comparison logic with position bias mitigation."""
        llm = self._get_llm()

        # Randomly assign positions to mitigate position bias
        swapped = random.random() > 0.5
        if swapped:
            resp_a, resp_b = response_reference, response_current
        else:
            resp_a, resp_b = response_current, response_reference

        prompt = PAIRWISE_PROMPT.format(
            query=query,
            response_a=resp_a[:2000],
            response_b=resp_b[:2000],
        )

        try:
            structured_llm = llm.with_structured_output(PairwiseJudgeOutput)
            result = structured_llm.invoke(prompt)
            winner = result.winner.upper().strip()
            reasoning = result.reasoning
            confidence = result.confidence
            dimension_wins = result.dimension_wins
        except Exception:
            # Fallback: text parsing
            try:
                result = llm.invoke(prompt)
                content = result.content if hasattr(result, "content") else str(result)
                from retune.utils.json_extract import extract_json_or_default
                parsed = extract_json_or_default(content, {})
                winner = str(parsed.get("winner", "tie")).upper().strip()
                reasoning = parsed.get("reasoning", content[:300])
                confidence = float(parsed.get("confidence", 0.5))
                dimension_wins = parsed.get("dimension_wins", {})
            except Exception as e:
                return EvalResult(
                    evaluator_name=self.name,
                    score=0.5,
                    reasoning=f"Pairwise comparison failed: {e}",
                    details={"mode": "error"},
                )

        # Un-swap: determine if 'current' won
        if swapped:
            # A was reference, B was current
            if winner == "A":
                current_won = False  # reference won
            elif winner == "B":
                current_won = True  # current won
            else:
                current_won = None  # tie
        else:
            # A was current, B was reference
            if winner == "A":
                current_won = True
            elif winner == "B":
                current_won = False
            else:
                current_won = None

        if current_won is True:
            score = min(0.5 + confidence * 0.5, 1.0)
        elif current_won is False:
            score = max(0.5 - confidence * 0.5, 0.0)
        else:
            score = 0.5

        return EvalResult(
            evaluator_name=self.name,
            score=round(score, 3),
            reasoning=reasoning,
            details={
                "mode": "pairwise",
                "winner": (
                    "current" if current_won
                    else ("reference" if current_won is False else "tie")
                ),
                "confidence": confidence,
                "swapped": swapped,
                "dimension_wins": dimension_wins,
            },
        )
