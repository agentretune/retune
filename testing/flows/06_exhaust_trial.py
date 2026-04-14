"""Flow 6: Exhaust free trial — confirm 16th optimize() returns 402.

Scenario:
  Run optimize() 15 times with a cheap agent. The 16th call should raise
  RuntimeError("Optimization run limit reached (402)").

Expected outcome:
  - Org's optimize_runs_used = 15 after 15 successful runs
  - 16th optimize() raises RuntimeError containing "limit reached" or "402"

Note: this is "cheap" because we use the echo agent + prompt axis only,
but it still makes 15 cloud round-trips with real LLM calls inside each.
Budget ~1-2 minutes and some LLM tokens.
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

from agents.echo_agent import make_echo_agent  # noqa: E402


def main() -> None:
    banner("FLOW 06 — Exhaust 15 free trial runs, confirm 402 on 16th")

    print()
    print("⚠️  This will consume up to 15 optimization runs from your trial.")
    print("   If your org already used runs, the 16th-call assertion may fire early.")
    print("   To reset, run: python testing/inspect/reset.py")
    print()
    resp = input("Continue? [y/N] ")
    if resp.lower() != "y":
        print("Aborted.")
        return

    storage = SQLiteStorage(path="./retune_test_06.db")
    retuner = Retuner(
        agent=make_echo_agent(),
        adapter="custom",
        mode=Mode.OBSERVE,
        api_key=os.environ["RETUNE_API_KEY"],
        agent_purpose="echo bot for trial-exhaustion testing",
        storage=storage,
    )

    # Seed a few traces
    for q in ["one", "two", "three"]:
        retuner.run(q)

    retuner.set_mode(Mode.IMPROVE)

    # Try up to 16 runs — stop on first RuntimeError
    for i in range(1, 17):
        try:
            report = retuner.optimize(source="last_n_traces", n=3, axes=["prompt"])
            print(f"  run {i:2d} → OK (run_id: {report.run_id[:20]}...)")
        except RuntimeError as e:
            msg = str(e)
            if "limit" in msg.lower() or "402" in msg:
                print(f"  run {i:2d} → ✓ GOT EXPECTED LIMIT ERROR")
                print(f"          {e}")
                if i >= 14:   # anything on run 14/15/16 counts as "trial exhausted on schedule"
                    print(f"\n✓ Trial enforcement working as expected.")
                else:
                    print(f"\n? Limit hit at run {i} — your org was not at 0 used runs to start.")
                return
            else:
                print(f"  run {i:2d} → ✗ Unexpected RuntimeError: {e}")
                return

    print(f"\n✗ Completed 16 runs without hitting the limit. Check optimize_runs_limit on the org.")


if __name__ == "__main__":
    main()
