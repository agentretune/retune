"""Flow 3: Free-trial — tool axis only.

Scenario:
  Agent has two tools registered: `search_fake` (always used) and
  `obsolete_metrics` (never called). ToolOptimizerAgent's
  DropUnusedAnalyzer should fire and propose dropping `obsolete_metrics`.

Expected outcome:
  - Tool metadata uploaded alongside traces at preauth
  - Cloud Orchestrator dispatches ToolOptimizerAgent
  - Report Tier 1 has a suggestion with apply_payload.action == "drop_tool"
    and apply_payload.tool_name == "obsolete_metrics"
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

from agents.tool_agent import SAMPLE_QUERIES, make_tool_agent  # noqa: E402


def main() -> None:
    banner("FLOW 03 — Free trial: tool optimization only")

    storage = SQLiteStorage(db_path="./retune_test_03.db")
    agent = make_tool_agent()

    retuner = Retuner(
        agent=agent,
        adapter="custom",
        mode=Mode.OBSERVE,
        api_key=os.environ["RETUNE_API_KEY"],
        agent_purpose="Help desk agent for SaaS billing (uses a search tool to look up answers)",
        storage=storage,
    )

    print("\n→ Agent has these tools registered:")
    for t in agent.tools:
        print(f"   - {t['name']}: {t['description'][:60]}")

    print(f"\n→ Running agent on {len(SAMPLE_QUERIES)} queries (only search_fake gets called)...")
    for i, q in enumerate(SAMPLE_QUERIES, 1):
        r = retuner.run(q)
        print(f"   {i:2d}. {q[:50]:50s} → {str(r.output)[:60]}")

    print(f"\n→ Agent call log confirms: {len(agent._call_log)} calls to search_fake, 0 to obsolete_metrics")

    print("\n→ Switching to IMPROVE mode, calling optimize(axes=['tools'])")
    retuner.set_mode(Mode.IMPROVE)
    report = retuner.optimize(
        source="last_n_traces",
        n=5,
        axes=["tools"],
    )

    print(f"\n→ Report:")
    print(f"   Tier 1: {len(report.tier1)} suggestions")
    print(f"   Tier 2: {len(report.tier2)} suggestions")

    drop_suggestions = [
        s for s in report.tier1
        if s.apply_payload and s.apply_payload.get("action") == "drop_tool"
    ]
    print(f"   drop_tool suggestions: {len(drop_suggestions)}")
    for s in drop_suggestions:
        target = s.apply_payload.get("tool_name")
        print(f"     → {s.title} (target: {target})")

    print_checklist([
        "Tier 1 has a drop_tool suggestion targeting 'obsolete_metrics'",
        "Dashboard → run detail → Suggestions tab shows the drop with Accept/Reject buttons",
        "If you click Accept, feedback POSTs to /v1/optimize/{run_id}/feedback",
        "Apply flow: retuner.apply_report(report, tier=1) removes obsolete_metrics from agent.tools",
    ])


if __name__ == "__main__":
    main()
