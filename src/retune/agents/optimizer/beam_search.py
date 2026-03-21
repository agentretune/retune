"""Beam Search APO — multi-round prompt optimization with verification rollouts.

Implements the Agent Lightning beam search approach:
1. Start with current prompt as the beam
2. For each round:
   a. Critique each candidate (generate textual gradient)
   b. Rewrite each candidate N times (branch_factor)
   c. Verify candidates via rollout (run agent with candidate config)
   d. Prune to top beam_width candidates
3. Return the best verified candidate
"""

from __future__ import annotations

import logging
from typing import Any

from retune.agents.optimizer.beam_config import BeamSearchConfig
from retune.core.models import (
    BeamCandidate,
    BeamSearchResult,
    ExecutionTrace,
    OptimizationConfig,
)
from retune.tools.builtin.credit_assigner import CreditAssignerTool
from retune.tools.builtin.prompt_analyzer import PromptAnalyzerTool
from retune.tools.builtin.prompt_rewriter import PromptRewriterTool
from retune.tools.builtin.rollout_runner import RolloutRunnerTool

logger = logging.getLogger(__name__)


class BeamSearchAPO:
    """Orchestrates Beam Search APO for prompt optimization.

    Flow per round:
    - For each prompt in the beam:
        - critique_fn(prompt, failures) → textual gradient
        - For i in range(branch_factor):
            - rewrite_fn(prompt, gradient) → candidate prompt
        - If verification_enabled:
            - rollout_runner(adapter, candidate, val_queries) → score
    - Prune to top beam_width candidates
    - Repeat for beam_rounds

    Uses tools directly (no Deep Agent orchestration) so it works
    regardless of whether deepagents is installed.
    """

    def __init__(
        self,
        config: BeamSearchConfig | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self._config = config or BeamSearchConfig()
        self._model = model
        self._cost_spent = 0.0
        self._prompt_analyzer = PromptAnalyzerTool()
        self._prompt_rewriter = PromptRewriterTool()
        self._rollout_runner = RolloutRunnerTool()
        self._credit_assigner = CreditAssignerTool()

    def search(
        self,
        current_prompt: str,
        failure_traces: list[dict[str, Any]],
        adapter: Any | None = None,
        evaluators: list[Any] | None = None,
        validation_queries: list[str] | None = None,
        current_config: OptimizationConfig | None = None,
    ) -> BeamSearchResult:
        """Run beam search APO to find the best prompt.

        Args:
            current_prompt: The starting system prompt
            failure_traces: Traces where the agent performed poorly
            adapter: Adapter for rollout verification (optional)
            evaluators: Evaluators for scoring rollouts (optional)
            validation_queries: Queries for rollout testing (optional)
            current_config: Current config for building rollout configs

        Returns:
            BeamSearchResult with the best prompt found
        """
        cfg = self._config
        beam: list[BeamCandidate] = [
            BeamCandidate(
                prompt=current_prompt,
                score=0.0,
                confidence=1.0,
                generation=0,
            )
        ]

        # Score the baseline
        baseline_score = self._estimate_baseline_score(failure_traces)
        beam[0].score = baseline_score

        all_candidates: list[BeamCandidate] = list(beam)

        # Configure rollout runner
        if adapter is not None:
            self._rollout_runner.set_adapter(adapter)
        if evaluators:
            self._rollout_runner.set_evaluators(evaluators)

        for round_num in range(1, cfg.beam_rounds + 1):
            if self._cost_spent >= cfg.cost_budget_usd:
                logger.info(f"Cost budget exhausted at round {round_num}")
                break

            logger.info(
                f"Beam Search Round {round_num}/{cfg.beam_rounds}: "
                f"{len(beam)} candidates in beam"
            )

            new_candidates: list[BeamCandidate] = []

            for candidate in beam:
                if self._cost_spent >= cfg.cost_budget_usd:
                    break

                # Step 1: Critique (generate textual gradient)
                gradient = self._critique(candidate.prompt, failure_traces)

                # Step 2: Rewrite (branch_factor times)
                for branch in range(cfg.branch_factor):
                    if self._cost_spent >= cfg.cost_budget_usd:
                        break

                    rewrite_result = self._rewrite(candidate.prompt, gradient)
                    new_prompt = rewrite_result.get("rewritten_prompt", "")
                    if not new_prompt or new_prompt == candidate.prompt:
                        continue

                    new_candidate = BeamCandidate(
                        prompt=new_prompt,
                        confidence=rewrite_result.get("confidence", 0.5),
                        generation=round_num,
                        parent_prompt=candidate.prompt[:100],
                        critique=gradient[:200],
                        changes_made=rewrite_result.get("changes_made", []),
                    )

                    # Step 3: Verify via rollout (if enabled and adapter available)
                    if cfg.verification_enabled and adapter is not None and validation_queries:
                        rollout_score = self._verify(
                            new_prompt, validation_queries, current_config
                        )
                        new_candidate.score = rollout_score
                        new_candidate.verified = True
                        new_candidate.verification_score = rollout_score
                    else:
                        # Estimate score from confidence
                        new_candidate.score = (
                            baseline_score + new_candidate.confidence * 0.2
                        )

                    new_candidates.append(new_candidate)
                    all_candidates.append(new_candidate)

            # Prune: keep top beam_width candidates
            combined = beam + new_candidates
            combined.sort(key=lambda c: c.score, reverse=True)
            beam = combined[: cfg.beam_width]

            logger.info(
                f"Round {round_num} complete: best score = {beam[0].score:.3f}, "
                f"candidates explored = {len(all_candidates)}"
            )

        # Final result
        best = beam[0] if beam else BeamCandidate(prompt=current_prompt)

        improvement = best.score - baseline_score

        return BeamSearchResult(
            best_prompt=best.prompt,
            best_score=best.score,
            baseline_score=baseline_score,
            improvement=improvement,
            candidates_explored=len(all_candidates),
            rounds_completed=min(cfg.beam_rounds, len(all_candidates)),
            total_cost_usd=self._cost_spent,
            beam_history=all_candidates,
            verified=best.verified,
        )

    def _estimate_baseline_score(self, failure_traces: list[dict[str, Any]]) -> float:
        """Estimate the baseline score from failure traces."""
        if not failure_traces:
            return 0.5

        scores = []
        for trace in failure_traces:
            eval_results = trace.get("eval_results", [])
            if eval_results:
                avg = sum(r.get("score", 0.5) for r in eval_results) / len(eval_results)
                scores.append(avg)

        return sum(scores) / len(scores) if scores else 0.5

    def _run_credit_assignment(self, trace: dict[str, Any]) -> dict[str, Any]:
        """Run credit assignment on a trace to identify bottleneck steps."""
        steps = trace.get("steps", [])
        response = str(trace.get("response", ""))
        eval_results = trace.get("eval_results", [])
        return self._credit_assigner.execute(
            steps=steps, response=response, eval_results=eval_results,
        )

    def _critique(self, prompt: str, failure_traces: list[dict[str, Any]]) -> str:
        """Generate textual gradient for a prompt.

        Uses credit assignment to identify WHICH steps failed and WHY,
        then connects those failures to specific prompt instructions
        (the credit-to-optimization bridge).
        """
        # Analyze prompt structure
        analysis = self._prompt_analyzer.execute(prompt=prompt)
        weaknesses = analysis.get("weaknesses", [])

        # Run credit assignment on failure traces (Agent Lightning bridge)
        bottleneck_summary = []
        for trace in failure_traces[:3]:
            credit = self._run_credit_assignment(trace)
            bottlenecks = credit.get("bottlenecks", [])
            if bottlenecks:
                top = bottlenecks[0]
                bottleneck_summary.append(
                    f"Query: '{trace.get('query', '')[:50]}' -- "
                    f"Bottleneck: [{top.get('step_type', '?')}] {top.get('name', '?')} "
                    f"(blame={top.get('blame_score', 0):.2f})"
                )

        # Build failure summary with credit data
        failure_examples = []
        for trace in failure_traces[:3]:
            failure_examples.append(
                f"Query: {trace.get('query', '')}\n"
                f"Response: {str(trace.get('response', ''))[:200]}\n"
                f"Scores: {[r.get('score', 0) for r in trace.get('eval_results', [])]}"
            )

        credit_text = ""
        if bottleneck_summary:
            credit_text = (
                "\n\nCREDIT ASSIGNMENT (which steps caused failures):\n"
                + "\n".join(f"- {b}" for b in bottleneck_summary)
                + "\n\nUse these bottlenecks to focus your critique on "
                "the prompt instructions that govern those specific step types."
            )

        try:
            from retune.core.llm import create_llm
            llm = create_llm(model=self._model, temperature=0)

            critique_prompt = (
                "You are performing APO (Automatic Prompt Optimization).\n\n"
                f"CURRENT SYSTEM PROMPT:\n\"\"\"\n{prompt}\n\"\"\"\n\n"
                f"PROMPT ANALYSIS:\nWeaknesses: {weaknesses}\n"
                f"Quality score: {analysis.get('quality_score', 0)}\n\n"
                f"FAILURE EXAMPLES:\n{'---'.join(failure_examples)}\n"
                f"{credit_text}\n\n"
                "Generate a SPECIFIC textual gradient (critique) explaining:\n"
                "1. Which prompt instructions caused or allowed these failures\n"
                "2. How the identified bottleneck steps connect to prompt gaps\n"
                "3. What needs to change and why\n"
                "4. Priority order of changes\n\n"
                "Be specific and actionable."
            )

            result = llm.invoke(critique_prompt)
            self._cost_spent += 0.001  # Approximate
            return result.content if hasattr(result, "content") else str(result)

        except Exception as e:
            logger.debug(f"LLM critique failed: {e}")
            # Heuristic critique from weaknesses + credit bottlenecks
            parts = []
            if weaknesses:
                parts.append("Prompt issues:\n" + "\n".join(f"- {w}" for w in weaknesses))
            if bottleneck_summary:
                parts.append("Bottleneck steps:\n" + "\n".join(f"- {b}" for b in bottleneck_summary))
            if parts:
                return "\n\n".join(parts)
            return "The prompt needs improvement but specific issues could not be determined."

    def _rewrite(self, prompt: str, gradient: str) -> dict[str, Any]:
        """Rewrite prompt using the textual gradient."""
        result = self._prompt_rewriter.execute(
            current_prompt=prompt,
            critique=gradient,
            style="conservative",
        )
        self._cost_spent += 0.001  # Approximate
        return result

    def _verify(
        self,
        prompt: str,
        validation_queries: list[str],
        current_config: OptimizationConfig | None,
    ) -> float:
        """Verify a candidate prompt via rollout."""
        config_dict = {}
        if current_config:
            config_dict = current_config.to_flat_dict()
        config_dict["system_prompt"] = prompt

        result = self._rollout_runner.execute(
            candidate_config=config_dict,
            validation_queries=validation_queries,
            max_queries=self._config.max_rollout_queries,
        )

        self._cost_spent += result.get("total_cost", 0.0)
        return result.get("avg_score", 0.0)
