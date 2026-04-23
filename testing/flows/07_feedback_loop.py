"""Flow 7: Feedback loop — rejected suggestion feeds the next run.

Scenario:
  1. Run optimize() → get a Tier 1 suggestion
  2. Submit feedback rejecting that suggestion with a reason
  3. Run optimize() again on the same org
  4. The Orchestrator should load the prior feedback and pass it to
     PromptOptimizerAgent.

Verification is indirect — feedback storage + retrieval is the mechanical
proof. Confirm via:
  - DB state: optimization_feedback table has the rejection
  - Uvicorn logs: "PromptOptimizerAgent initialized with N prior feedback entries"
"""
from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
import json as _json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import banner, load_env, print_checklist, require_env  # noqa: E402

load_env()
require_env("RETUNE_API_KEY", "RETUNE_CLOUD_BASE_URL")

from retune import Mode, Retuner  # noqa: E402
from retune.storage import SQLiteStorage  # noqa: E402

from agents.bad_prompt_agent import SAMPLE_QUERIES, make_bad_prompt_agent  # noqa: E402


def _submit_feedback(run_id: str, suggestion_id: str, accepted: bool, comment: str) -> None:
    """POST to /v1/optimize/{run_id}/feedback as the SDK user."""
    url = f"{os.environ['RETUNE_CLOUD_BASE_URL']}/api/v1/optimize/{run_id}/feedback"
    payload = {
        "suggestion_id": suggestion_id,
        "tier": 1,
        "accepted": accepted,
        "comment": comment,
    }
    req = urllib.request.Request(
        url,
        data=_json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['RETUNE_API_KEY']}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return _json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"   ✗ feedback POST failed: {e.code} {e.reason}")
        return None


def main() -> None:
    banner("FLOW 07 — Feedback loop: reject then re-run")

    storage = SQLiteStorage(db_path="./retune_test_07.db")
    agent = make_bad_prompt_agent()

    retuner = Retuner(
        agent=agent,
        adapter="custom",
        mode=Mode.OBSERVE,
        api_key=os.environ["RETUNE_API_KEY"],
        agent_purpose="Customer support bot — testing feedback loop",
        storage=storage,
    )

    print(f"\n→ Phase A: seed traces + first optimize")
    for q in SAMPLE_QUERIES[:5]:
        retuner.run(q)

    retuner.set_mode(Mode.IMPROVE)
    report1 = retuner.optimize(source="last_n_traces", n=5, axes=["prompt"])
    print(f"   report1.run_id = {report1.run_id}")
    print(f"   Tier 1 count: {len(report1.tier1)}")

    if not report1.tier1:
        print("   ? No Tier 1 suggestion to reject. Cannot continue test.")
        return

    target = report1.tier1[0]
    print(f"\n→ Phase B: reject Tier 1 suggestion")
    print(f"   Target: {target.title}")
    _submit_feedback(
        run_id=report1.run_id,
        suggestion_id=target.title,
        accepted=False,
        comment="The tone still feels too formal for our product voice.",
    )
    print(f"   ✓ Rejection submitted")

    print(f"\n→ Phase C: seed a few more traces + second optimize")
    retuner.set_mode(Mode.OBSERVE)
    for q in SAMPLE_QUERIES[5:]:
        retuner.run(q)

    retuner.set_mode(Mode.IMPROVE)
    report2 = retuner.optimize(source="last_n_traces", n=5, axes=["prompt"])
    print(f"   report2.run_id = {report2.run_id}")

    print_checklist([
        "Uvicorn logs between run1 and run2 include:\n      'PromptOptimizerAgent initialized with N prior feedback entries'",
        "DB: optimization_feedback table has 1 row with accepted=false",
        "Dashboard run detail → Feedback tab shows the rejection",
        "report2's Tier 1 prompt is different from report1's (the rejected one)",
    ])


if __name__ == "__main__":
    main()
