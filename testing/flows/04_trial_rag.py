"""Flow 4: Free-trial — RAG axis only.

Scenario:
  LangChain RAG agent with chunk_size=1500 (too large). RAGOptimizerAgent
  should analyze retrieval patterns and propose Tier 2 parameter sweep
  suggestions — most reliably a chunk_sweep to 800.

Expected outcome:
  - retrieval_config (k, chunk_size, strategy) uploaded at preauth
  - Cloud Orchestrator dispatches RAGOptimizerAgent
  - Report has Tier 2 suggestions with apply_payload.action == "apply_retrieval_override"
  - Possibly Tier 3 conceptual suggestions (e.g. "consider hybrid retrieval")
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
from retune.storage import SQLiteStorage  # noqa: E402

from agents.rag_agent import SAMPLE_QUERIES, make_rag_agent  # noqa: E402


def main() -> None:
    banner("FLOW 04 — Free trial: RAG optimization only")

    storage = SQLiteStorage(path="./retune_test_04.db")
    agent = make_rag_agent(k=5, chunk_size=1500)   # intentionally oversized

    retuner = Retuner(
        agent=agent,
        adapter="custom",
        mode=Mode.OBSERVE,
        api_key=os.environ["RETUNE_API_KEY"],
        agent_purpose="Billing FAQ RAG bot — retrieves from policy docs",
        storage=storage,
    )

    print(f"\n→ Agent retrieval config: {agent.retrieval_config}")
    print(f"→ Running agent on {len(SAMPLE_QUERIES)} queries...")
    for i, q in enumerate(SAMPLE_QUERIES, 1):
        r = retuner.run(q)
        print(f"   {i:2d}. {q[:50]:50s} → {str(r.output)[:60]}")

    print("\n→ Switching to IMPROVE mode, calling optimize(axes=['rag'])")
    retuner.set_mode(Mode.IMPROVE)
    report = retuner.optimize(
        source="last_n_traces",
        n=10,
        axes=["rag"],
    )

    print(f"\n→ Report:")
    print(f"   Tier 2: {len(report.tier2)} suggestions")
    print(f"   Tier 3: {len(report.tier3)} conceptual")

    rag_tier2 = [s for s in report.tier2 if s.axis == "rag"]
    for s in rag_tier2:
        action = (s.apply_payload or {}).get("action")
        print(f"\n   [T2] {s.title}")
        print(f"        action: {action}")
        print(f"        overrides: { {k: v for k, v in (s.apply_payload or {}).items() if k != 'action'} }")

    for s in report.tier3:
        if s.axis == "rag":
            print(f"\n   [T3] {s.title}")

    print_checklist([
        "Tier 2 has at least one suggestion with action == 'apply_retrieval_override'",
        "chunk_size appears in at least one apply_payload (because baseline was 1500)",
        "Dashboard → run detail → Suggestions tab shows the suggestions with 'Copy snippet' buttons",
        "Pareto tab: shows scatter of cost × quality × latency across explored candidates",
    ])


if __name__ == "__main__":
    main()
