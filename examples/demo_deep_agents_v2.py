"""Demo: Deep Agents v2 -- Full Agent Lightning + LangChain Deep Agents Architecture.

This demo shows the complete pipeline:
1. EvaluatorDeepAgent with subagent delegation
2. OptimizerDeepAgent with Beam Search APO
3. Graceful degradation (works without deepagents/LLM)

Run: python examples/demo_deep_agents_v2.py
"""

from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO, format="%(name)s -- %(message)s")
logger = logging.getLogger("demo")


def demo_models():
    """Show the new models: Span, BeamCandidate, BeamSearchResult."""
    from retune.core.enums import StepType
    from retune.core.models import BeamCandidate, BeamSearchResult, Span

    print("\n" + "=" * 60)
    print("1. NEW MODELS")
    print("=" * 60)

    # Span -- links to Step, stores contribution scores
    span = Span(
        step_id="step-001",
        step_type=StepType.RETRIEVAL,
        name="retrieve_docs",
        contribution_score=-0.4,
        is_bottleneck=True,
        reasoning="Retrieved irrelevant documents -> downstream generation failed",
    )
    print(f"\nSpan: {span.name} (contribution={span.contribution_score}, bottleneck={span.is_bottleneck})")

    # BeamCandidate -- a candidate in the beam search
    candidate = BeamCandidate(
        prompt="You are a precise research assistant. Always cite sources.",
        score=0.82,
        confidence=0.85,
        generation=2,
        parent_prompt="You are a helpful assistant.",
        changes_made=["Added precision focus", "Added citation requirement"],
        verified=True,
        verification_score=0.80,
    )
    print(f"BeamCandidate: score={candidate.score}, verified={candidate.verified}, gen={candidate.generation}")

    # BeamSearchResult -- overall search outcome
    result = BeamSearchResult(
        best_prompt=candidate.prompt,
        best_score=0.82,
        baseline_score=0.55,
        improvement=0.27,
        candidates_explored=12,
        rounds_completed=2,
        total_cost_usd=0.04,
        beam_history=[candidate],
        verified=True,
    )
    print(f"BeamSearchResult: improvement=+{result.improvement:.2f}, "
          f"candidates={result.candidates_explored}, cost=${result.total_cost_usd:.2f}")


def demo_beam_config():
    """Show BeamSearchConfig options."""
    from retune.agents.optimizer.beam_config import BeamSearchConfig

    print("\n" + "=" * 60)
    print("2. BEAM SEARCH CONFIG")
    print("=" * 60)

    # Default config
    default = BeamSearchConfig()
    print(f"\nDefault: width={default.beam_width}, branches={default.branch_factor}, "
          f"rounds={default.beam_rounds}, budget=${default.cost_budget_usd:.2f}")

    # Aggressive search
    aggressive = BeamSearchConfig(
        beam_width=4,
        branch_factor=3,
        beam_rounds=3,
        cost_budget_usd=1.00,
    )
    print(f"Aggressive: width={aggressive.beam_width}, branches={aggressive.branch_factor}, "
          f"rounds={aggressive.beam_rounds}, budget=${aggressive.cost_budget_usd:.2f}")


def demo_new_tools():
    """Show the new tools: PromptRewriter, RolloutRunner, GradientAggregator."""
    from retune.tools.builtin.gradient_aggregator import GradientAggregatorTool
    from retune.tools.builtin.prompt_rewriter import PromptRewriterTool

    print("\n" + "=" * 60)
    print("3. NEW TOOLS")
    print("=" * 60)

    # PromptRewriter (heuristic mode -- no LLM needed)
    rewriter = PromptRewriterTool()
    result = rewriter._heuristic_rewrite(
        "Answer questions about science.",
        "Missing role definition and step-by-step reasoning instructions",
    )
    print(f"\nPromptRewriter (heuristic):")
    print(f"  Original: 'Answer questions about science.'")
    print(f"  Rewritten: '{result['rewritten_prompt'][:100]}...'")
    print(f"  Changes: {result['changes_made']}")

    # GradientAggregator (heuristic mode)
    aggregator = GradientAggregatorTool()
    result = aggregator._heuristic_aggregate([
        "The prompt lacks a clear role definition -- the agent doesn't know who it is",
        "No examples provided -- the agent needs demonstrations of expected behavior",
        "Missing output format constraints -- responses are inconsistent",
        "Role is vague, needs specific domain expertise mention",
    ])
    print(f"\nGradientAggregator (heuristic):")
    print(f"  Themes detected: {result['themes']}")


def demo_subagent_definitions():
    """Show the subagent definitions for evaluator and optimizer."""
    from retune.agents.evaluator.subagents.definitions import (
        get_all_evaluator_subagent_definitions,
    )
    from retune.agents.optimizer.subagents.definitions import (
        get_all_optimizer_subagent_definitions,
    )

    print("\n" + "=" * 60)
    print("4. SUBAGENT DEFINITIONS")
    print("=" * 60)

    print("\nEvaluator Subagents:")
    for d in get_all_evaluator_subagent_definitions():
        tools = [t.name for t in d["tools"]]
        print(f"  {d['name']}: {d['description'][:60]}... tools={tools}")

    print("\nOptimizer Subagents:")
    for d in get_all_optimizer_subagent_definitions():
        tools = [t.name for t in d["tools"]]
        print(f"  {d['name']}: {d['description'][:60]}... tools={tools}")


def demo_evaluator_heuristic():
    """Demo the evaluator in heuristic fallback mode (no LLM needed)."""
    from retune.agents.evaluator.agent import EvaluatorDeepAgent
    from retune.core.enums import StepType
    from retune.core.models import EvalResult, ExecutionTrace, Step

    print("\n" + "=" * 60)
    print("5. EVALUATOR DEEP AGENT (heuristic fallback)")
    print("=" * 60)

    trace = ExecutionTrace(
        query="What is the capital of France?",
        response="The capital of France is Paris.",
        steps=[
            Step(
                step_type=StepType.RETRIEVAL,
                name="retrieve_docs",
                input_data={"query": "capital of France"},
                output_data={"documents": [{"content": "Paris is the capital of France"}]},
            ),
            Step(
                step_type=StepType.REASONING,
                name="plan_answer",
                input_data={"context": "retrieved docs"},
                output_data={"plan": "Answer with capital city from sources"},
            ),
            Step(
                step_type=StepType.LLM_CALL,
                name="generate_answer",
                input_data={"prompt": "Answer based on context"},
                output_data={"response": "The capital of France is Paris."},
            ),
        ],
    )

    evaluator = EvaluatorDeepAgent()
    evaluator._use_deep_agents = False  # Force heuristic

    # Use heuristic evaluate directly to avoid needing LangGraph
    result = evaluator._heuristic_evaluate(trace)

    print(f"\nQuery: '{trace.query}'")
    print(f"Response: '{trace.response}'")
    print(f"Steps: {len(trace.steps)}")
    print(f"Score: {result.score}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Mode: {result.details.get('mode', 'unknown')}")


def demo_beam_search_mock():
    """Demo beam search APO with mocked LLM calls."""
    from unittest.mock import patch

    from retune.agents.optimizer.beam_config import BeamSearchConfig
    from retune.agents.optimizer.beam_search import BeamSearchAPO
    from retune.core.models import OptimizationConfig

    print("\n" + "=" * 60)
    print("6. BEAM SEARCH APO (mocked LLM)")
    print("=" * 60)

    cfg = BeamSearchConfig(beam_rounds=2, branch_factor=2, beam_width=2)
    apo = BeamSearchAPO(config=cfg)

    failure_traces = [
        {
            "query": "Explain quantum computing",
            "response": "It uses qubits",
            "eval_results": [{"score": 0.3}],
        },
        {
            "query": "What causes rain?",
            "response": "Water falls",
            "eval_results": [{"score": 0.4}],
        },
    ]

    rewrite_counter = [0]

    def mock_critique(prompt, traces):
        return "Missing: detailed explanations, source citations, structured format"

    def mock_rewrite(prompt, gradient):
        rewrite_counter[0] += 1
        return {
            "rewritten_prompt": (
                f"You are a knowledgeable science educator (v{rewrite_counter[0]}). "
                "Provide detailed, well-structured explanations. "
                "Always cite your reasoning. Use step-by-step format."
            ),
            "changes_made": [f"Iteration {rewrite_counter[0]}"],
            "confidence": 0.6 + rewrite_counter[0] * 0.05,
        }

    with patch.object(apo, "_critique", side_effect=mock_critique), \
         patch.object(apo, "_rewrite", side_effect=mock_rewrite):
        result = apo.search(
            current_prompt="You are helpful. Answer questions.",
            failure_traces=failure_traces,
        )

    print(f"\nBaseline score: {result.baseline_score:.3f}")
    print(f"Best score: {result.best_score:.3f}")
    print(f"Improvement: +{result.improvement:.3f}")
    print(f"Candidates explored: {result.candidates_explored}")
    print(f"Rounds completed: {result.rounds_completed}")
    print(f"Cost: ${result.total_cost_usd:.4f}")
    print(f"Best prompt: '{result.best_prompt[:80]}...'")


def demo_architecture_overview():
    """Print the full architecture overview."""
    print("\n" + "=" * 60)
    print("ARCHITECTURE OVERVIEW: Deep Agents v2")
    print("=" * 60)

    print("""
    EVALUATOR (Deep Agent with subagents):
      Main Agent -> plans evaluation -> delegates:
        |-- trace-analyzer    -> step analysis, timing, tokens
        |-- credit-assigner   -> Agent Lightning blame attribution
        |-- tool-auditor      -> tool usage efficiency
        `-- hallucination-detector -> response grounding

    OPTIMIZER (Deep Agent + Beam Search APO):
      Main Agent -> plans optimization -> delegates:
        |-- prompt-critic     -> APO textual gradient
        |-- prompt-rewriter   -> APO prompt rewriting (xbranch_factor)
        |-- config-tuner      -> parameter optimization
        `-- tool-curator      -> tool enable/disable suggestions

    BEAM SEARCH APO:
      Round 1: [current_prompt] -> critique -> rewritexN -> verify -> prune
      Round 2: [survivors]      -> critique -> rewritexN -> verify -> prune
      Result:  best verified candidate -> Suggestion

    EXECUTION MODES:
      deepagents installed -> Full Deep Agent with TodoList + SubAgent middleware
      langgraph installed  -> LangGraph StateGraph fallback
      neither installed    -> Pure heuristic fallback (no LLM needed)
    """)


if __name__ == "__main__":
    demo_architecture_overview()
    demo_models()
    demo_beam_config()
    demo_new_tools()
    demo_subagent_definitions()
    demo_evaluator_heuristic()
    demo_beam_search_mock()

    print("\n" + "=" * 60)
    print("Demo complete! All components working.")
    print("=" * 60)
