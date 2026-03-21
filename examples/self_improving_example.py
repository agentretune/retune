"""Example: Full self-improvement loop (the fan regulator in action).

Shows how to:
1. Run in OBSERVE mode to collect traces
2. Switch to EVALUATE to score performance
3. Run evaluation dataset
4. Switch to IMPROVE to get optimization suggestions
5. Apply improvements and verify
"""

from retune import Retuner, Mode, OptimizationConfig


def my_rag_pipeline(query: str) -> str:
    """Simulated RAG pipeline — replace with your real one."""
    # Simulate different quality responses
    if "machine learning" in query.lower():
        return "Machine learning is a subset of AI that enables systems to learn from data."
    elif "rag" in query.lower():
        return "RAG stands for Retrieval-Augmented Generation."
    else:
        return "I'm not sure about that topic."


def main():
    # --- Phase 1: OBSERVE ---
    print("=== Phase 1: OBSERVE ===")
    wrapped = Retuner(
        agent=my_rag_pipeline,
        adapter="custom",
        mode=Mode.OBSERVE,
        config=OptimizationConfig(top_k=3, temperature=0.7),
    )

    queries = [
        "What is machine learning?",
        "How does RAG work?",
        "Tell me about quantum physics",
    ]

    for q in queries:
        response = wrapped.run(q)
        print(f"  Q: {q} -> A: {response.output[:80]}")

    print(f"  Traces collected: {len(wrapped.get_traces())}")

    # --- Phase 2: EVALUATE ---
    # (Requires LLM evaluator — using latency evaluator as demo)
    print("\n=== Phase 2: EVALUATE ===")
    from retune.evaluators.latency import LatencyEvaluator
    from retune.evaluators.retrieval import RetrievalEvaluator

    wrapped_eval = Retuner(
        agent=my_rag_pipeline,
        adapter="custom",
        mode=Mode.EVALUATE,
        evaluators=[LatencyEvaluator(), RetrievalEvaluator()],
        config=OptimizationConfig(top_k=3, temperature=0.7),
    )

    for q in queries:
        response = wrapped_eval.run(q)
        scores = {r.evaluator_name: r.score for r in response.eval_results}
        print(f"  Q: {q} -> Scores: {scores}")

    summary = wrapped_eval.get_eval_summary()
    print(f"  Eval summary: {summary['scores']}")

    # --- Phase 3: IMPROVE ---
    print("\n=== Phase 3: IMPROVE ===")
    wrapped_eval.set_mode(Mode.IMPROVE)

    response = wrapped_eval.run("What is deep learning?")
    if response.suggestions:
        print("  Suggestions:")
        for s in response.suggestions:
            print(f"    {s.param_name}: {s.old_value} -> {s.new_value} ({s.reasoning})")
    else:
        print("  No suggestions (system is performing well)")

    # --- Phase 4: Turn OFF and run with improved config ---
    print("\n=== Phase 4: OFF (production) ===")
    wrapped_eval.set_mode(Mode.OFF)
    response = wrapped_eval.run("What is machine learning?")
    print(f"  Running in production mode: {response.output}")
    print(f"  Current version: v{wrapped_eval.version}")


if __name__ == "__main__":
    main()
