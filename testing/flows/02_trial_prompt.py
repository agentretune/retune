"""Flow 2: Free-trial — prompt axis only.

Scenario:
  Developer wraps an agent with a deliberately vague system prompt
  ("be helpful"). After ~10 traces, they hit IMPROVE and request
  axes=["prompt"] only. The cloud optimizer should propose a clearer
  prompt via Beam Search APO + LLM rewriter.

Expected outcome:
  - Trace upload at preauth succeeds
  - Cloud Orchestrator dispatches PromptOptimizerAgent (only)
  - Rewriter LLM (user's choice) is actually called — check your Anthropic/OpenAI usage
  - Report has at least 1 Tier 1 prompt rewrite
  - optimize_runs_used incremented by 1

Manual verification:
  - Check uvicorn logs: "PromptOptimizerAgent initialized..."
  - Check LLM provider dashboard: tokens consumed
  - Query DB: SELECT optimize_runs_used FROM organizations
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

from agents.bad_prompt_agent import SAMPLE_QUERIES, make_bad_prompt_agent  # noqa: E402


def main() -> None:
    banner("FLOW 02 — Free trial: prompt optimization only")

    # Point SDK at local cloud
    os.environ["RETUNE_CLOUD_BASE_URL"] = os.environ["RETUNE_CLOUD_BASE_URL"]

    storage = SQLiteStorage(path="./retune_test_02.db")
    agent = make_bad_prompt_agent()

    retuner = Retuner(
        agent=agent,
        adapter="custom",
        mode=Mode.OBSERVE,
        api_key=os.environ["RETUNE_API_KEY"],
        agent_purpose="Customer support bot for SaaS billing questions",
        success_criteria="Accurate, concise answers citing policy when applicable",
        evaluators=[
            CostEvaluator(),
            LatencyEvaluator(),
            LLMJudgeEvaluator(criteria="helpful and accurate"),
        ],
        storage=storage,
    )

    print(f"\n→ Baseline system prompt: {agent.system_prompt!r}")
    print(f"→ Running agent on {len(SAMPLE_QUERIES)} queries (this hits your LLM)...")
    for i, q in enumerate(SAMPLE_QUERIES, 1):
        r = retuner.run(q)
        print(f"   {i:2d}. {q[:50]:50s} → {str(r.output)[:60]}")

    print("\n→ Switching to IMPROVE mode, calling optimize(axes=['prompt'])")
    retuner.set_mode(Mode.IMPROVE)
    report = retuner.optimize(
        source="last_n_traces",
        n=10,
        axes=["prompt"],
        rewriter_llm="claude-3-5-sonnet-20241022" if os.environ.get("ANTHROPIC_API_KEY") else "gpt-4o-mini",
    )

    print(f"\n→ Report received:")
    print(f"   run_id: {report.run_id}")
    print(f"   Tier 1 suggestions: {len(report.tier1)}")
    print(f"   Tier 2 suggestions: {len(report.tier2)}")
    print(f"   Tier 3 suggestions: {len(report.tier3)}")

    for s in report.tier1:
        print(f"\n   [T1] {s.title}")
        if s.description:
            print(f"        {s.description[:200]}")

    print_checklist([
        f"Report run_id {report.run_id} appears in dashboard at http://localhost:5173/runs",
        "Tier 1 has at least one prompt rewrite (not just baseline)",
        "Rewriter LLM actually ran — check your LLM provider's usage dashboard",
        "DB state: python testing/inspect/db_state.py shows runs_used incremented",
        "Uvicorn logs show 'OptimizerOrchestrator' and 'PromptOptimizerAgent' activity",
    ])


if __name__ == "__main__":
    main()
