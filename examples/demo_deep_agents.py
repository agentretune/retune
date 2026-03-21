"""
DEMO: Deep Agent Evaluation & Optimization (LLM-powered, real intelligence)

This demo uses the REAL deep agents — LangGraph-based evaluator and optimizer
that use actual LLM calls for:

1. EVALUATOR DEEP AGENT (LangGraph supervisor):
   - TraceAnalyzer: step-by-step execution breakdown
   - CreditAssigner: Agent Lightning per-step blame attribution
   - ToolAuditor: tool usage accuracy check
   - HallucinationDetector: LLM checks response against sources
   - Synthesizer: LLM aggregates all analyses into final score

2. OPTIMIZER DEEP AGENT (LangGraph planner):
   - APO (Automatic Prompt Optimization from Agent Lightning):
     Step 1: EVALUATE current prompt against failures
     Step 2: CRITIQUE - generate "textual gradient" feedback
     Step 3: REWRITE - produce improved prompt
   - ConfigTuner: parameter search based on eval scores
   - ToolCurator: tool enable/disable based on usage patterns

Requires: OPENAI_API_KEY in .env
    pip install retune[agents]
"""

import os
import sys
import random
import time
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: Set OPENAI_API_KEY in .env file")
    print("Example: OPENAI_API_KEY=sk-...")
    sys.exit(1)

from retune import Retuner, Mode, OptimizationConfig
from retune.core.enums import StepType
from retune.core.models import ExecutionTrace, Step, TokenUsage
from retune.adapters.base import BaseAdapter
from retune.agents.evaluator.agent import EvaluatorDeepAgent
from retune.agents.optimizer.agent import OptimizerDeepAgent
from retune.storage.sqlite_storage import SQLiteStorage


# =============================================================================
# SIMULATED RAG AGENT (same as before, but now evaluated by REAL LLM)
# =============================================================================

KNOWLEDGE_BASE = {
    "machine learning": [
        "Machine learning is a branch of AI that enables systems to learn patterns from data.",
        "Supervised learning uses labeled data; unsupervised learning finds hidden patterns.",
        "Common algorithms include decision trees, random forests, SVMs, and neural networks.",
    ],
    "deep learning": [
        "Deep learning uses neural networks with multiple layers to learn hierarchical representations.",
        "Transformers revolutionized NLP with the self-attention mechanism.",
        "GPU acceleration is essential for training large models.",
    ],
    "rag": [
        "RAG combines retrieval with generation to produce grounded answers.",
        "RAG reduces hallucinations by basing responses on retrieved documents.",
        "Embedding models convert text to vectors for semantic search.",
    ],
}


class SimulatedAgent(BaseAdapter):
    """Agent whose quality depends on config — evaluated by REAL LLM."""

    def __init__(self, agent: Any, **kwargs: Any) -> None:
        super().__init__(agent=agent, **kwargs)
        self._config = OptimizationConfig()

    def run(self, query: str, config: OptimizationConfig | None = None, **kwargs: Any) -> ExecutionTrace:
        if config:
            self._config = config

        top_k = self._config.top_k or 1
        temperature = self._config.temperature if self._config.temperature is not None else 0.9
        system_prompt = self._config.system_prompt or ""

        started_at = datetime.now(timezone.utc)
        steps = []

        # Retrieval
        query_lower = query.lower()
        matched_topic = None
        for topic in KNOWLEDGE_BASE:
            if topic in query_lower:
                matched_topic = topic
                break

        docs = []
        if matched_topic:
            docs = KNOWLEDGE_BASE[matched_topic][:min(top_k, 3)]
        else:
            docs = [list(KNOWLEDGE_BASE.values())[0][0]]

        ret_start = datetime.now(timezone.utc)
        time.sleep(0.05)
        steps.append(Step(
            step_type=StepType.RETRIEVAL,
            name="vector_retriever",
            input_data={"query": query},
            output_data={
                "documents": [{"content": d, "metadata": {}} for d in docs],
                "num_docs": len(docs),
            },
            started_at=ret_start,
            ended_at=datetime.now(timezone.utc),
        ))

        # Generate response
        llm_start = datetime.now(timezone.utc)
        time.sleep(0.05)

        if len(docs) >= 2 and temperature < 0.5:
            response = " ".join(docs[:2])
        elif len(docs) == 1:
            response = docs[0]
            if temperature > 0.7 and random.random() < 0.5:
                response += " This is also related to quantum computing advancements."
        else:
            response = " ".join(docs[:2])

        steps.append(Step(
            step_type=StepType.LLM_CALL,
            name="gpt-4o-mini",
            input_data={"system_prompt": system_prompt[:100], "query": query},
            output_data={"response": response},
            token_usage=TokenUsage(
                prompt_tokens=100 + len(docs) * 30,
                completion_tokens=len(response.split()) * 2,
                total_tokens=100 + len(docs) * 30 + len(response.split()) * 2,
            ),
            started_at=llm_start,
            ended_at=datetime.now(timezone.utc),
        ))

        return ExecutionTrace(
            query=query,
            response=response,
            steps=steps,
            config_snapshot=self._config.to_flat_dict(),
            started_at=started_at,
            ended_at=datetime.now(timezone.utc),
        )

    def get_config(self) -> OptimizationConfig:
        return self._config.model_copy()

    def apply_config(self, config: OptimizationConfig) -> None:
        self._config = config


# =============================================================================
# THE DEMO
# =============================================================================

def print_header(text: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")


def main():
    random.seed(42)

    queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "Explain RAG and how it reduces hallucinations",
    ]

    import tempfile
    db_path = os.path.join(tempfile.mkdtemp(), "deep_demo.db")

    # =====================================================
    # PHASE 1: Setup with Deep Agents
    # =====================================================
    print_header("SETUP: Using LLM-powered Deep Agents")
    print("  Evaluator: EvaluatorDeepAgent (LangGraph supervisor)")
    print("    -> TraceAnalyzer, CreditAssigner, HallucinationDetector, Synthesizer")
    print("  Optimizer: OptimizerDeepAgent (LangGraph planner)")
    print("    -> APO (Evaluate->Critique->Rewrite), ConfigTuner, ToolCurator")

    bad_config = OptimizationConfig(
        top_k=1,
        temperature=0.9,
        system_prompt="You are a helpful assistant.",
    )

    print(f"\n  Starting config (bad):")
    print(f"    top_k={bad_config.top_k}, temperature={bad_config.temperature}")
    print(f"    prompt: \"{bad_config.system_prompt}\"")

    # Create deep agents
    evaluator = EvaluatorDeepAgent(model="gpt-4o-mini")
    optimizer = OptimizerDeepAgent(model="gpt-4o-mini")

    adapter = SimulatedAgent(agent=lambda q: q)

    wrapped = Retuner(
        agent=lambda q: q,
        adapter=adapter,
        mode=Mode.EVALUATE,
        config=bad_config,
        evaluators=[evaluator],
        optimizer=optimizer,
        storage=SQLiteStorage(db_path),
    )

    # =====================================================
    # PHASE 2: Deep Evaluation
    # =====================================================
    print_header("PHASE 1: DEEP EVALUATION (LLM-powered)")
    print("  Each query goes through the full LangGraph evaluation pipeline...")

    baseline_scores = []
    for q in queries:
        print(f"\n  Q: {q}")
        response = wrapped.run(q)
        answer = (response.output or "")[:100]
        if len(response.output or "") > 100:
            answer += "..."
        print(f"  A: {answer}")

        for r in response.eval_results:
            print(f"  [{r.evaluator_name}] Score: {r.score:.2f}")
            print(f"    Reasoning: {r.reasoning[:200]}")
            if r.details:
                detail_str = ", ".join(
                    f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in r.details.items()
                    if k not in ("bottlenecks", "error") and isinstance(v, (int, float))
                )
                if detail_str:
                    print(f"    Details: {detail_str}")
            baseline_scores.append(r.score)

    avg_baseline = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0
    print(f"\n  >> Average baseline score: {avg_baseline:.2f}")

    # =====================================================
    # PHASE 3: Deep Optimization with APO
    # =====================================================
    print_header("PHASE 2: DEEP OPTIMIZATION (APO + ConfigTuner)")
    print("  Running the Optimizer Deep Agent...")
    print("  - APO: Evaluate prompt -> Critique -> Rewrite")
    print("  - ConfigTuner: Analyze scores, search parameter space")

    wrapped.set_mode(Mode.IMPROVE)
    wrapped._auto_improve = True

    response = wrapped.run("What is machine learning?")

    if response.suggestions:
        print(f"\n  >> {len(response.suggestions)} SUGGESTIONS from Deep Optimizer:")
        for i, s in enumerate(response.suggestions, 1):
            old_display = s.old_value
            new_display = s.new_value
            # Truncate long prompts for display
            if isinstance(old_display, str) and len(old_display) > 80:
                old_display = old_display[:80] + "..."
            if isinstance(new_display, str) and len(new_display) > 80:
                new_display = new_display[:80] + "..."

            print(f"\n  {i}. [{s.category.upper():>6s}] {s.param_name}")
            print(f"     Confidence: {s.confidence:.0%}")
            print(f"     Before: {old_display}")
            print(f"     After:  {new_display}")
            print(f"     Why: {s.reasoning[:300]}")

        # Show the full rewritten prompt if APO produced one
        prompt_suggestions = [s for s in response.suggestions if s.param_name == "system_prompt"]
        if prompt_suggestions:
            print(f"\n  >> APO REWRITTEN PROMPT:")
            print(f"  ---")
            new_prompt = prompt_suggestions[0].new_value
            for line in str(new_prompt).split("\n"):
                print(f"  {line}")
            print(f"  ---")

    print(f"\n  Config AFTER optimization (v{wrapped.version}):")
    new_config = wrapped.get_config()
    print(f"    top_k={new_config.top_k}, temperature={new_config.temperature}")
    prompt_preview = (new_config.system_prompt or "")[:100]
    if len(new_config.system_prompt or "") > 100:
        prompt_preview += "..."
    print(f"    prompt: \"{prompt_preview}\"")

    # =====================================================
    # PHASE 4: Re-evaluate with improved config
    # =====================================================
    print_header("PHASE 3: RE-EVALUATE with improved config")

    wrapped.set_mode(Mode.EVALUATE)
    improved_scores = []
    for q in queries:
        print(f"\n  Q: {q}")
        response = wrapped.run(q)
        answer = (response.output or "")[:100]
        if len(response.output or "") > 100:
            answer += "..."
        print(f"  A: {answer}")
        for r in response.eval_results:
            print(f"  [{r.evaluator_name}] Score: {r.score:.2f} - {r.reasoning[:150]}")
            improved_scores.append(r.score)

    avg_improved = sum(improved_scores) / len(improved_scores) if improved_scores else 0

    # =====================================================
    # SUMMARY
    # =====================================================
    print_header("SUMMARY")
    print(f"\n  Average score BEFORE: {avg_baseline:.2f}")
    print(f"  Average score AFTER:  {avg_improved:.2f}")
    delta = avg_improved - avg_baseline
    sign = "+" if delta > 0 else ""
    print(f"  Change: {sign}{delta:.2f}")

    print(f"\n  This was powered by REAL LLM calls, not hardcoded rules.")
    print(f"  - Evaluator used GPT-4o-mini to detect hallucinations")
    print(f"  - Optimizer used GPT-4o-mini for APO prompt rewriting")
    print(f"  - Credit assignment identified which steps caused failures")
    print()


if __name__ == "__main__":
    main()
