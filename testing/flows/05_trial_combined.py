"""Flow 5: Free-trial — all 3 axes in one run.

Scenario:
  RAG agent with tools and vague system prompt — optimize all 3 axes at once.
  All three subagents dispatch in the cloud.

Expected outcome:
  - Report has Tier 1 from prompt+tools, Tier 2 from rag+tools, Tier 3 from all
  - DB state: one run, with traces+tools+retrieval_config all stored
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import banner, load_env, print_checklist, require_env  # noqa: E402

load_env()
require_env("RETUNE_API_KEY", "RETUNE_CLOUD_BASE_URL")

from retune import Mode, Retuner  # noqa: E402
from retune.evaluators import CostEvaluator, LatencyEvaluator, LLMJudgeEvaluator  # noqa: E402
from retune.storage import SQLiteStorage  # noqa: E402

from agents.rag_agent import SAMPLE_QUERIES, make_rag_agent  # noqa: E402


def main() -> None:
    banner("FLOW 05 — Free trial: combined prompt + tools + rag")

    storage = SQLiteStorage(path="./retune_test_05.db")
    agent = make_rag_agent(k=5, chunk_size=1500)

    retuner = Retuner(
        agent=agent,
        adapter="custom",
        mode=Mode.OBSERVE,
        api_key=os.environ["RETUNE_API_KEY"],
        agent_purpose="Billing support RAG bot — answers policy questions from the corpus",
        success_criteria="Concise, cited, no hallucinated details",
        evaluators=[
            CostEvaluator(),
            LatencyEvaluator(),
            LLMJudgeEvaluator(criteria="helpful, concise, accurate"),
        ],
        storage=storage,
    )

    print(f"\n→ Running agent on {len(SAMPLE_QUERIES)} queries...")
    for i, q in enumerate(SAMPLE_QUERIES, 1):
        r = retuner.run(q)
        print(f"   {i:2d}. {q[:50]:50s} → {str(r.output)[:50]}")

    print("\n→ optimize(axes=['prompt','tools','rag'])")
    retuner.set_mode(Mode.IMPROVE)
    report = retuner.optimize(
        source="last_n_traces",
        n=10,
        axes=["prompt", "tools", "rag"],
        rewriter_llm="claude-3-5-sonnet-20241022" if os.environ.get("ANTHROPIC_API_KEY") else "gpt-4o-mini",
    )

    print(f"\n→ Report summary:")
    print(f"   improvement: {report.summary.get('improvement_pct', 0):.1f}%")
    print(f"   Tier 1: {len(report.tier1)}  Tier 2: {len(report.tier2)}  Tier 3: {len(report.tier3)}")

    axes_seen = {s.axis for s in (report.tier1 + report.tier2 + report.tier3)}
    print(f"   axes represented: {sorted(axes_seen)}")

    print_checklist([
        "Report has suggestions across multiple axes (prompt, tools, rag expected)",
        "Dashboard shows the run with a tab each for Overview / Suggestions / Pareto / Feedback",
        "Pareto scatter has at least 3 points (baseline + variants)",
        "DB query: testing/inspect/db_state.py — one run with status=completed",
    ])


if __name__ == "__main__":
    main()
