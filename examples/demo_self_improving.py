"""
DEMO: Watch Retune diagnose and fix a bad RAG pipeline in real-time.

This example simulates a RAG pipeline with REAL, measurable problems:
- Only 1 document retrieved (top_k=1) -> incomplete answers
- High temperature (0.9) -> hallucinations in responses
- No reranking -> irrelevant docs mixed in
- Slow retrieval

The self-improvement loop detects each issue, suggests fixes, applies them,
and you can see the scores go UP across iterations.

No API keys needed - runs entirely locally.
"""

import time
import random
from datetime import datetime, timezone
from typing import Any

from retune import Retuner, Mode, OptimizationConfig
from retune.core.enums import StepType
from retune.core.models import (
    EvalResult,
    ExecutionTrace,
    Step,
    Suggestion,
    TokenUsage,
)
from retune.evaluators.base import BaseEvaluator
from retune.optimizers.base import BaseOptimizer
from retune.adapters.base import BaseAdapter


# =============================================================================
# 1. SIMULATED KNOWLEDGE BASE
# =============================================================================

KNOWLEDGE_BASE = {
    "machine learning": [
        "Machine learning is a branch of AI that enables systems to learn patterns from data without being explicitly programmed.",
        "Supervised learning uses labeled data to train models that can predict outcomes for new inputs.",
        "Common ML algorithms include decision trees, random forests, SVMs, and neural networks.",
        "Feature engineering is the process of selecting and transforming variables to improve model performance.",
        "Overfitting occurs when a model memorizes training data but fails to generalize to new unseen data.",
    ],
    "deep learning": [
        "Deep learning uses artificial neural networks with multiple hidden layers to learn hierarchical representations of data.",
        "CNNs (Convolutional Neural Networks) are specifically designed for image and spatial data processing.",
        "Transformers revolutionized NLP by introducing the self-attention mechanism, enabling parallel processing of sequences.",
        "Backpropagation is the core algorithm used to train neural networks by computing gradients layer by layer.",
        "GPU acceleration is essential for training large deep learning models efficiently due to parallel computation.",
    ],
    "rag": [
        "RAG (Retrieval-Augmented Generation) combines a retrieval system with a generative LLM to produce grounded answers.",
        "RAG reduces hallucinations by forcing the LLM to base its responses on retrieved factual documents.",
        "Chunking strategy significantly impacts RAG quality - too small loses context, too large adds noise.",
        "Embedding models convert text into dense vectors for semantic similarity search in vector databases.",
        "Reranking retrieved documents with a cross-encoder can boost retrieval precision by 20-40%.",
    ],
    "agents": [
        "AI agents are autonomous systems that perceive their environment and take goal-directed actions.",
        "ReAct (Reasoning + Acting) is a paradigm that interleaves chain-of-thought reasoning with tool use.",
        "Multi-agent systems coordinate multiple specialized agents to solve complex tasks collaboratively.",
        "Tool use allows agents to call APIs, query databases, run code, and interact with external systems.",
        "Agent memory systems store past interactions and learnings to improve future decision-making.",
    ],
}


# =============================================================================
# 2. SIMULATED RAG PIPELINE (quality depends on config!)
# =============================================================================

class SimulatedRAGAdapter(BaseAdapter):
    """A simulated RAG pipeline whose output quality directly depends on config.

    Bad config (top_k=1, temp=0.9, no reranker) -> bad results.
    Good config (top_k=4, temp=0.2, reranker=True) -> good results.
    """

    def __init__(self, agent: Any, **kwargs: Any) -> None:
        super().__init__(agent=agent, **kwargs)
        self._config = OptimizationConfig()

    def run(self, query: str, config: OptimizationConfig | None = None, **kwargs: Any) -> ExecutionTrace:
        if config:
            self._config = config

        top_k = self._config.top_k or 1
        temperature = self._config.temperature if self._config.temperature is not None else 0.9
        use_reranker = self._config.use_reranker or False

        started_at = datetime.now(timezone.utc)
        steps = []

        # --- Step 1: Retrieval ---
        retrieval_start = datetime.now(timezone.utc)

        # Simulate latency that scales with top_k
        latency_s = 0.1 + (top_k * 0.05)
        time.sleep(latency_s)

        # Find matching topic
        query_lower = query.lower()
        matched_topic = None
        for topic in KNOWLEDGE_BASE:
            if topic in query_lower:
                matched_topic = topic
                break

        retrieved_docs = []
        if matched_topic:
            all_docs = KNOWLEDGE_BASE[matched_topic]
            # top_k controls how many RELEVANT docs we retrieve
            n_relevant = min(top_k, len(all_docs))
            relevant_docs = all_docs[:n_relevant]

            # Without reranker, noise docs get mixed in (pollutes context)
            if not use_reranker and top_k >= 2:
                noise_topics = [t for t in KNOWLEDGE_BASE if t != matched_topic]
                noise_doc = random.choice(KNOWLEDGE_BASE[random.choice(noise_topics)])
                # Replace last relevant doc with noise
                relevant_docs[-1] = noise_doc

            retrieved_docs = [
                {"content": doc, "metadata": {"source": matched_topic}}
                for doc in relevant_docs
            ]
        else:
            # No match -> retrieve irrelevant fallback
            fallback_topic = random.choice(list(KNOWLEDGE_BASE.keys()))
            retrieved_docs = [
                {"content": KNOWLEDGE_BASE[fallback_topic][0],
                 "metadata": {"source": "fallback"}}
            ]

        retrieval_end = datetime.now(timezone.utc)
        steps.append(Step(
            step_type=StepType.RETRIEVAL,
            name="vector_retriever",
            input_data={"query": query},
            output_data={"documents": retrieved_docs, "num_docs": len(retrieved_docs)},
            started_at=retrieval_start,
            ended_at=retrieval_end,
        ))

        # --- Step 2: LLM Generation ---
        llm_start = datetime.now(timezone.utc)
        time.sleep(0.1)

        context_parts = [doc["content"] for doc in retrieved_docs]

        # Generate response - QUALITY DEPENDS ON CONFIG
        if not matched_topic:
            response = "I don't have enough information to answer this question accurately."
        elif n_relevant == 1:
            # Only 1 doc -> very thin answer
            base = context_parts[0]
            if temperature > 0.6 and random.random() < 0.5:
                # High temp -> hallucinate extra content
                response = base + " Furthermore, this is closely related to quantum computing and blockchain technology."
            else:
                response = base
        elif not use_reranker:
            # Multiple docs but noise mixed in -> confused answer
            if temperature > 0.6 and random.random() < 0.4:
                response = context_parts[0] + " " + context_parts[-1]  # Mixes relevant + noise
                response += " This topic is also fundamental to cryptocurrency mining."
            else:
                response = " ".join(context_parts[:2])  # Some noise included
        else:
            # Reranker ON + multiple docs = best quality
            response = " ".join(context_parts[:min(3, len(context_parts))])

        prompt_tokens = sum(len(d.split()) for d in context_parts) * 3 + 80
        completion_tokens = len(response.split()) * 2
        total_tokens = prompt_tokens + completion_tokens

        llm_end = datetime.now(timezone.utc)
        steps.append(Step(
            step_type=StepType.LLM_CALL,
            name="gpt-4o-mini",
            input_data={"prompt": f"Context: ...{len(context_parts)} docs...\nQuery: {query}"},
            output_data={"response": response},
            token_usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            started_at=llm_start,
            ended_at=llm_end,
        ))

        ended_at = datetime.now(timezone.utc)
        return ExecutionTrace(
            query=query,
            response=response,
            steps=steps,
            config_snapshot=self._config.to_flat_dict(),
            started_at=started_at,
            ended_at=ended_at,
        )

    def get_config(self) -> OptimizationConfig:
        return self._config.model_copy()

    def apply_config(self, config: OptimizationConfig) -> None:
        self._config = config


# =============================================================================
# 3. EVALUATORS (simulate real quality checks without API calls)
# =============================================================================

class AnswerQualityEvaluator(BaseEvaluator):
    """Scores answer quality: length, relevance, hallucination, confidence."""
    name = "answer_quality"

    TOPIC_KEYWORDS = {
        "machine learning": ["learning", "data", "model", "algorithm", "training", "supervised", "patterns"],
        "deep learning": ["neural", "network", "layers", "deep", "cnn", "transformer", "backpropagation"],
        "rag": ["retrieval", "generation", "document", "embedding", "chunk", "hallucination", "grounded"],
        "agents": ["agent", "tool", "action", "environment", "reasoning", "autonomous", "goal"],
    }

    HALLUCINATION_MARKERS = [
        "quantum computing", "blockchain", "cryptocurrency",
        "mining", "qubits",
    ]

    def evaluate(self, trace: ExecutionTrace) -> EvalResult:
        response = (trace.response or "").lower()
        query = (trace.query or "").lower()
        word_count = len(response.split())

        score = 0.0
        reasons = []

        # 1. Completeness (0-0.30) - longer, more detailed = better
        if word_count >= 40:
            score += 0.30
            reasons.append(f"Comprehensive ({word_count} words)")
        elif word_count >= 25:
            score += 0.20
            reasons.append(f"Adequate detail ({word_count} words)")
        elif word_count >= 15:
            score += 0.10
            reasons.append(f"Brief ({word_count} words)")
        else:
            reasons.append(f"Too short ({word_count} words)")

        # 2. Relevance (0-0.30) - topic keywords present
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            if topic in query:
                hits = sum(1 for kw in keywords if kw in response)
                rel_score = min(hits / 4, 1.0) * 0.30
                score += rel_score
                reasons.append(f"Relevance: {hits}/{len(keywords)} keywords")
                break
        else:
            score += 0.10
            reasons.append("Unknown topic")

        # 3. No hallucinations (0-0.25)
        hallucinations = [m for m in self.HALLUCINATION_MARKERS if m in response]
        if not hallucinations:
            score += 0.25
            reasons.append("Clean - no hallucinations")
        else:
            reasons.append(f"HALLUCINATION: {', '.join(hallucinations)}")

        # 4. Confidence (0-0.15)
        if "don't have enough" in response or "not sure" in response:
            reasons.append("Low confidence")
        else:
            score += 0.15
            reasons.append("Confident answer")

        final_score = round(min(score, 1.0), 2)
        return EvalResult(
            evaluator_name=self.name,
            score=final_score,
            reasoning="; ".join(reasons),
            details={"correctness": final_score},
        )


class DocCoverageEvaluator(BaseEvaluator):
    """Scores retrieval quality based on doc count and coverage."""
    name = "retrieval"

    def evaluate(self, trace: ExecutionTrace) -> EvalResult:
        retrieval_steps = [s for s in trace.steps if s.step_type == StepType.RETRIEVAL]
        if not retrieval_steps:
            return EvalResult(evaluator_name=self.name, score=0.5,
                              reasoning="No retrieval steps found",
                              details={"retrieval": 0.5})

        total_docs = 0
        for step in retrieval_steps:
            total_docs += step.output_data.get("num_docs", 0)

        # Score: 1 doc = 0.3, 2 docs = 0.5, 3 docs = 0.7, 4+ docs = 0.9-1.0
        if total_docs <= 1:
            score = 0.30
            reasoning = f"Only {total_docs} doc retrieved - insufficient coverage"
        elif total_docs == 2:
            score = 0.55
            reasoning = f"{total_docs} docs retrieved - moderate coverage"
        elif total_docs == 3:
            score = 0.75
            reasoning = f"{total_docs} docs retrieved - good coverage"
        else:
            score = min(0.85 + (total_docs - 4) * 0.05, 1.0)
            reasoning = f"{total_docs} docs retrieved - excellent coverage"

        # Check if docs are actually referenced in response
        response_lower = (trace.response or "").lower()
        for step in retrieval_steps:
            for doc in step.output_data.get("documents", []):
                content = doc.get("content", "").lower()
                words = [w for w in content.split()[:8] if len(w) > 4]
                if any(w in response_lower for w in words):
                    score = min(score + 0.10, 1.0)
                    break

        return EvalResult(
            evaluator_name=self.name,
            score=round(score, 2),
            reasoning=reasoning,
            details={"retrieval": round(score, 2), "total_docs": total_docs},
        )


# =============================================================================
# 4. CUSTOM OPTIMIZER (more aggressive rules for the demo)
# =============================================================================

class DemoOptimizer(BaseOptimizer):
    """Optimizer tuned for this demo - catches the deliberately-bad config."""
    name = "demo_optimizer"

    def suggest(self, traces: list[ExecutionTrace], current_config: OptimizationConfig) -> list[Suggestion]:
        if not traces:
            return []

        suggestions = []

        # Aggregate scores
        score_lists: dict[str, list[float]] = {}
        for trace in traces:
            for r in trace.eval_results:
                score_lists.setdefault(r.evaluator_name, []).append(r.score)
                for key, val in r.details.items():
                    if isinstance(val, (int, float)):
                        score_lists.setdefault(key, []).append(float(val))

        avg_scores = {k: sum(v) / len(v) for k, v in score_lists.items()}

        # Rule 1: Low retrieval coverage -> increase top_k
        retrieval_score = avg_scores.get("retrieval", 1.0)
        current_top_k = current_config.top_k or 1
        if retrieval_score < 0.6 and current_top_k < 5:
            suggestions.append(Suggestion(
                param_name="top_k",
                old_value=current_top_k,
                new_value=min(current_top_k + 3, 5),
                reasoning=f"Retrieval score is {retrieval_score:.2f} (below 0.6). Retrieving more documents will improve coverage and answer completeness.",
                confidence=0.85,
                category="rag",
            ))

        # Rule 2: Hallucinations detected -> lower temperature
        correctness_score = avg_scores.get("correctness", 1.0)
        quality_score = avg_scores.get("answer_quality", 1.0)
        current_temp = current_config.temperature if current_config.temperature is not None else 0.9
        if (correctness_score < 0.7 or quality_score < 0.7) and current_temp > 0.3:
            suggestions.append(Suggestion(
                param_name="temperature",
                old_value=current_temp,
                new_value=0.2,
                reasoning=f"Answer quality is {quality_score:.2f}. High temperature ({current_temp}) causes hallucinations. Lowering to 0.2 for more deterministic outputs.",
                confidence=0.90,
                category="agent",
            ))

        # Rule 3: No reranker -> enable it
        if not current_config.use_reranker and retrieval_score < 0.8:
            suggestions.append(Suggestion(
                param_name="use_reranker",
                old_value=False,
                new_value=True,
                reasoning=f"Retrieval score is {retrieval_score:.2f}. A cross-encoder reranker filters out irrelevant documents, typically improving precision by 20-40%.",
                confidence=0.80,
                category="rag",
            ))

        return suggestions


# =============================================================================
# 5. THE DEMO
# =============================================================================

def print_header(text: str) -> None:
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}")


def print_config(config: OptimizationConfig, indent: str = "  ") -> None:
    print(f"{indent}top_k={config.top_k}, temperature={config.temperature}, "
          f"reranker={config.use_reranker}, search={config.search_type or 'similarity'}")


def run_eval_round(wrapped: Retuner, queries: list[str], round_name: str) -> dict[str, float]:
    """Run queries and return average scores per evaluator."""
    print(f"\n  --- {round_name} ---")
    all_scores: dict[str, list[float]] = {}

    for q in queries:
        response = wrapped.run(q)
        scores_str = ", ".join(f"{r.evaluator_name}={r.score:.2f}" for r in response.eval_results)
        answer = (response.output or "")[:90]
        if len(response.output or "") > 90:
            answer += "..."
        print(f"    Q: {q}")
        print(f"       A: {answer}")
        print(f"       Scores: [{scores_str}]")

        for r in response.eval_results:
            all_scores.setdefault(r.evaluator_name, []).append(r.score)

    return {k: sum(v) / len(v) for k, v in all_scores.items()}


def print_scoreboard(scores: dict[str, float], baseline: dict[str, float] | None = None) -> None:
    for name, avg in scores.items():
        bar_len = int(avg * 30)
        bar = "#" * bar_len + " " * (30 - bar_len)
        if baseline and name in baseline:
            delta = avg - baseline[name]
            sign = "+" if delta > 0 else ""
            status = f"({sign}{delta:.2f})"
        else:
            status = "GOOD" if avg >= 0.7 else "NEEDS WORK" if avg >= 0.4 else "POOR"
        print(f"     {name:20s} [{bar}] {avg:.2f}  {status}")


def main():
    random.seed(42)

    queries = [
        "What is machine learning?",
        "Explain deep learning and neural networks",
        "How does RAG reduce hallucinations?",
        "What are AI agents and how do they work?",
        "Tell me about transformers in deep learning",
    ]

    # Setup storage
    import tempfile, os
    db_path = os.path.join(tempfile.mkdtemp(), "demo.db")
    from retune.storage.sqlite_storage import SQLiteStorage
    from retune.evaluators.latency import LatencyEvaluator

    # =====================================================
    # PHASE 1: BAD CONFIG - Observe baseline
    # =====================================================
    print_header("PHASE 1: OBSERVE with DELIBERATELY BAD config")

    bad_config = OptimizationConfig(
        top_k=1,             # Too few docs!
        temperature=0.9,     # Too random - causes hallucinations!
        use_reranker=False,  # No filtering of irrelevant docs!
        search_type="similarity",
    )
    print("  Starting with a bad config:")
    print_config(bad_config)

    adapter = SimulatedRAGAdapter(agent=lambda q: q)

    wrapped = Retuner(
        agent=lambda q: q,
        adapter=adapter,
        mode=Mode.OBSERVE,
        config=bad_config,
        evaluators=[
            AnswerQualityEvaluator(),
            DocCoverageEvaluator(),
            LatencyEvaluator(),
        ],
        optimizer=DemoOptimizer(),
        storage=SQLiteStorage(db_path),
    )

    # Observe phase - just collect traces
    for q in queries:
        wrapped.run(q)
    print(f"\n  Traces collected: {len(wrapped.get_traces())}")

    # =====================================================
    # PHASE 2: EVALUATE - See the problems
    # =====================================================
    print_header("PHASE 2: EVALUATE - Measuring baseline quality")
    wrapped.set_mode(Mode.EVALUATE)

    baseline_scores = run_eval_round(wrapped, queries, "Baseline Evaluation")

    print(f"\n  >> BASELINE SCORES:")
    print_scoreboard(baseline_scores)

    # =====================================================
    # PHASE 3: IMPROVE - Diagnose and fix
    # =====================================================
    print_header("PHASE 3: IMPROVE - Self-improvement loop")
    wrapped.set_mode(Mode.IMPROVE)
    wrapped._auto_improve = True

    print(f"\n  Config BEFORE (v{wrapped.version}):")
    print_config(wrapped.get_config(), indent="    ")

    # Run improvement - this triggers diagnosis + auto-apply
    response = wrapped.run("What is machine learning?")

    if response.suggestions:
        print(f"\n  >> {len(response.suggestions)} IMPROVEMENT SUGGESTIONS:")
        for i, s in enumerate(response.suggestions, 1):
            conf_bar = "#" * int(s.confidence * 10) + " " * (10 - int(s.confidence * 10))
            print(f"\n  {i}. [{s.category.upper()}] {s.param_name}: {s.old_value} -> {s.new_value}")
            print(f"     Confidence: [{conf_bar}] {s.confidence:.0%}")
            print(f"     Reason: {s.reasoning}")
    else:
        print("\n  No suggestions generated.")

    print(f"\n  Config AFTER auto-improvement (v{wrapped.version}):")
    print_config(wrapped.get_config(), indent="    ")

    # =====================================================
    # PHASE 4: RE-EVALUATE with improved config
    # =====================================================
    print_header("PHASE 4: RE-EVALUATE - Did it actually improve?")
    wrapped.set_mode(Mode.EVALUATE)

    improved_scores = run_eval_round(wrapped, queries, "Post-Improvement Evaluation")

    print(f"\n  >> IMPROVED SCORES (vs baseline):")
    print_scoreboard(improved_scores, baseline=baseline_scores)

    # =====================================================
    # FINAL SUMMARY
    # =====================================================
    print_header("FINAL SUMMARY")

    print(f"\n  {'Metric':<20s} {'Before':>8s} {'After':>8s} {'Change':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")

    total_b, total_a, count = 0, 0, 0
    for name in baseline_scores:
        b = baseline_scores[name]
        a = improved_scores.get(name, b)
        d = a - b
        sign = "+" if d > 0 else ""
        marker = " <<" if d > 0.05 else ""
        print(f"  {name:<20s} {b:>8.2f} {a:>8.2f} {sign}{d:>7.2f}{marker}")
        total_b += b; total_a += a; count += 1

    if count:
        avg_b, avg_a = total_b / count, total_a / count
        d = avg_a - avg_b
        sign = "+" if d > 0 else ""
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
        print(f"  {'OVERALL':<20s} {avg_b:>8.2f} {avg_a:>8.2f} {sign}{d:>7.2f}")

    # Config diff
    history = wrapped.get_improvement_history()
    if history:
        print(f"\n  Config changes (v1 -> v{wrapped.version}):")
        for entry in history:
            for s in entry["suggestions"]:
                print(f"    {s['param_name']}: {s['old_value']} -> {s['new_value']}")

    print(f"\n  The system diagnosed its own problems and fixed them automatically!")
    print()


if __name__ == "__main__":
    main()
