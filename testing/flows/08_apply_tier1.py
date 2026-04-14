"""Flow 8: Apply a Tier 1 suggestion, confirm agent state mutates.

Scenario:
  - Run optimize on the tool_agent (has a never-called tool)
  - Tier 1 has a drop_tool suggestion
  - Call retuner.apply_report(report, tier=1)
  - Verify agent.tools no longer contains the dropped tool
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
    banner("FLOW 08 — Apply Tier 1 drop_tool, confirm agent.tools mutates")

    storage = SQLiteStorage(db_path="./retune_test_08.db")
    agent = make_tool_agent()

    print(f"\n→ Before: agent.tools = {[t['name'] for t in agent.tools]}")

    retuner = Retuner(
        agent=agent,
        adapter="custom",
        mode=Mode.OBSERVE,
        api_key=os.environ["RETUNE_API_KEY"],
        agent_purpose="Help desk for billing — tests apply flow",
        storage=storage,
    )

    for q in SAMPLE_QUERIES:
        retuner.run(q)

    retuner.set_mode(Mode.IMPROVE)
    report = retuner.optimize(source="last_n_traces", n=5, axes=["tools"])

    drop_suggs = [
        s for s in report.tier1
        if s.apply_payload and s.apply_payload.get("action") == "drop_tool"
    ]
    if not drop_suggs:
        print("   ? No drop_tool suggestion in Tier 1. Cannot test apply.")
        return

    target_name = drop_suggs[0].apply_payload["tool_name"]
    print(f"\n→ Tier 1 suggests dropping tool: {target_name!r}")

    print("\n→ Calling retuner.apply_report(report, tier=1)")
    retuner.apply_report(report, tier=1)

    after_names = [t["name"] if isinstance(t, dict) else getattr(t, "name", str(t)) for t in agent.tools]
    print(f"   After: agent.tools = {after_names}")

    if target_name not in after_names:
        print(f"\n   ✓ {target_name} successfully dropped from agent.tools")
    else:
        print(f"\n   ✗ {target_name} STILL in agent.tools — apply failed")
        sys.exit(1)

    print_checklist([
        "Tier 1 drop_tool suggestion was discovered by ToolOptimizerAgent",
        "apply_report removed the target from agent.tools in-memory",
        "(Optional) Dashboard → Suggestions tab: Accept button on drop_tool triggers same flow",
    ])


if __name__ == "__main__":
    main()
