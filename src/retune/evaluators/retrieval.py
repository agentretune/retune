"""Retrieval evaluator — scores the quality of retrieved documents."""

from __future__ import annotations

from retune.core.enums import StepType
from retune.core.models import EvalResult, ExecutionTrace
from retune.evaluators.base import BaseEvaluator


class RetrievalEvaluator(BaseEvaluator):
    """Evaluates retrieval quality based on step metadata.

    Checks:
    - Were documents retrieved?
    - How many documents?
    - Were they used in the final response?
    """

    name = "retrieval"

    def evaluate(self, trace: ExecutionTrace) -> EvalResult:
        retrieval_steps = [s for s in trace.steps if s.step_type == StepType.RETRIEVAL]

        if not retrieval_steps:
            return EvalResult(
                evaluator_name=self.name,
                score=1.0,  # No retrieval needed = not penalized
                reasoning="No retrieval steps found — agent may not use RAG.",
                details={"has_retrieval": False},
            )

        total_docs = 0
        non_empty_retrievals = 0

        for step in retrieval_steps:
            num_docs = step.output_data.get("num_docs", 0)
            total_docs += num_docs
            if num_docs > 0:
                non_empty_retrievals += 1

        # Score based on whether retrieval returned results
        if len(retrieval_steps) == 0:
            retrieval_success_rate = 0.0
        else:
            retrieval_success_rate = non_empty_retrievals / len(retrieval_steps)

        # Check if response references retrieved content
        response_str = str(trace.response).lower()
        docs_referenced = False
        for step in retrieval_steps:
            docs = step.output_data.get("documents", [])
            for doc in docs:
                content = doc.get("content", "").lower()
                if content and any(
                    word in response_str for word in content.split()[:10] if len(word) > 4
                ):
                    docs_referenced = True
                    break

        # Composite score
        score = 0.5 * retrieval_success_rate
        if docs_referenced:
            score += 0.5
        elif retrieval_success_rate > 0:
            score += 0.2  # Retrieved but unclear if used

        return EvalResult(
            evaluator_name=self.name,
            score=min(score, 1.0),
            reasoning=(
                f"Retrieved {total_docs} docs across {len(retrieval_steps)} retrieval steps. "
                f"Success rate: {retrieval_success_rate:.0%}. "
                f"Docs referenced in response: {docs_referenced}."
            ),
            details={
                "has_retrieval": True,
                "total_docs": total_docs,
                "retrieval_steps": len(retrieval_steps),
                "success_rate": retrieval_success_rate,
                "docs_referenced": docs_referenced,
            },
        )
