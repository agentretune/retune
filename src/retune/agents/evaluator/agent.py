"""Deep Evaluator — calls cloud API for premium evaluation.

With API key: sends trace to agentretune.com, gets deep evaluation back.
Without API key: falls back to local heuristic evaluation.
The actual deep evaluation logic (LangGraph subagents, hallucination
detection, credit assignment, LLM synthesis) runs on the server.
"""

from __future__ import annotations

import json
import logging
from urllib.error import URLError
from urllib.request import Request, urlopen

from retune.core.models import EvalResult, ExecutionTrace
from retune.evaluators.base import BaseEvaluator
from retune.tools.builtin.credit_assigner import CreditAssignerTool
from retune.tools.builtin.trace_reader import TraceReaderTool

logger = logging.getLogger(__name__)


class EvaluatorDeepAgent(BaseEvaluator):
    """Deep evaluator — premium cloud evaluation with local fallback.

    With API key (free or paid):
    - Sends trace to cloud API for deep evaluation
    - Server runs LangGraph supervisor with 4 subagents
    - Returns detailed scores (correctness, completeness, grounding, etc.)
    - Counts against 15 free deep operations

    Without API key:
    - Heuristic evaluation using TraceReader + CreditAssigner
    - No LLM calls, no usage counted
    """

    name = "deep_evaluator"

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._model = model

    def evaluate(self, trace: ExecutionTrace) -> EvalResult:
        # Try cloud evaluation first
        try:
            from retune.config import settings
            if settings.api_key:
                result = self._cloud_evaluate(trace, settings.api_key, settings.cloud_base_url)
                if result:
                    return result
        except Exception as e:
            logger.debug(f"Cloud evaluation failed: {e}")

        # Fallback to local heuristic
        return self._heuristic_evaluate(trace)

    def _cloud_evaluate(
        self, trace: ExecutionTrace, api_key: str, base_url: str
    ) -> EvalResult | None:
        """Send trace to cloud API for deep evaluation."""
        url = f"{base_url}/api/v1/hosted/judge"
        payload = json.dumps({
            "query": trace.query,
            "response": str(trace.response)[:3000],
            "expected": trace.metadata.get("expected_answer"),
            "evaluators": ["deep_evaluator"],
            "model": self._model,
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "retune-sdk/0.1.0",
        }

        try:
            req = Request(url, data=payload, headers=headers, method="POST")
            with urlopen(req, timeout=30.0) as resp:
                data = json.loads(resp.read())
                results = data.get("eval_results", [])
                if results:
                    r = results[0]
                    return EvalResult(
                        evaluator_name=self.name,
                        score=r.get("score", 0.5),
                        reasoning=r.get("reasoning", ""),
                        details=r.get("details", {}),
                    )
        except URLError as e:
            logger.debug(f"Cloud eval request failed: {e}")
        except Exception as e:
            logger.debug(f"Cloud eval parse failed: {e}")

        return None

    def _heuristic_evaluate(self, trace: ExecutionTrace) -> EvalResult:
        """Local heuristic fallback (no LLM, no usage counted)."""
        tool = TraceReaderTool()
        analysis = tool.execute(trace=trace.model_dump(mode="json"))

        credit_tool = CreditAssignerTool()
        credit = credit_tool.execute(
            steps=[s.model_dump(mode="json") for s in trace.steps],
            response=str(trace.response),
            eval_results=[r.model_dump(mode="json") for r in trace.eval_results],
        )

        scores = []
        if analysis.get("has_reasoning"):
            scores.append(0.8)
        else:
            scores.append(0.4)
        scores.append(credit.get("overall_score", 0.5))

        final_score = sum(scores) / len(scores) if scores else 0.5

        return EvalResult(
            evaluator_name=self.name,
            score=round(final_score, 2),
            reasoning=(
                f"Heuristic evaluation: {analysis.get('total_steps', 0)} steps, "
                f"credit score {credit.get('overall_score', 0.5):.2f}"
            ),
            details={
                "trace_analysis": analysis,
                "credit_assignment": credit,
                "mode": "local_heuristic",
            },
        )
