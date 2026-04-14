"""Flow 1: Pure-SDK free mode — no cloud, no API key.

Scenario:
  A developer installs `retune` and wraps an agent. They want local
  observability + evaluation without signing up for anything.

Expected outcome:
  - Traces accumulate in ./retune_test.db (local SQLite)
  - EVALUATE mode populates eval_results from cost + latency evaluators
  - IMPROVE mode FAILS with a clear error (no api_key → no cloud optimizer)
  - `python -m retune dashboard` can serve the traces locally
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import banner, load_env, print_checklist  # noqa: E402

load_env()

from retune import Mode, Retuner  # noqa: E402
from retune.evaluators import CostEvaluator, LatencyEvaluator  # noqa: E402

from agents.echo_agent import make_echo_agent  # noqa: E402


def main() -> None:
    banner("FLOW 01 — Pure-SDK free mode (no cloud)")

    # Use a test-specific SQLite so we don't pollute the default retune.db
    from retune.storage import SQLiteStorage
    storage = SQLiteStorage(path="./retune_test_01.db")

    retuner = Retuner(
        agent=make_echo_agent(),
        adapter="custom",
        mode=Mode.OBSERVE,
        evaluators=[CostEvaluator(), LatencyEvaluator()],
        storage=storage,
        # NO api_key — fully local
    )

    # Phase A: OBSERVE mode — just capture traces
    print("\n→ Phase A: OBSERVE mode, 5 queries")
    for q in ["Hello", "What is ML?", "Explain vectors", "Run A/B", "Why test?"]:
        r = retuner.run(q)
        print(f"   {q!r:30s} → {str(r.output)[:40]!r}")

    # Phase B: EVALUATE mode — scores get attached
    print("\n→ Phase B: EVALUATE mode, 3 queries")
    retuner.set_mode(Mode.EVALUATE)
    for q in ["Score this", "And this", "One more"]:
        r = retuner.run(q)
        scores = {e.evaluator_name: round(e.score, 4) for e in (r.eval_results or [])}
        print(f"   {q!r:30s} → scores: {scores}")

    # Phase C: IMPROVE mode — must fail without api_key
    print("\n→ Phase C: IMPROVE mode without api_key (expected: clean error)")
    retuner.set_mode(Mode.IMPROVE)
    # Note: just switching to IMPROVE doesn't error; calling optimize() does
    try:
        # optimize requires agent_purpose — first rebuild the retuner
        from retune import Retuner as _R
        retuner2 = _R(
            agent=make_echo_agent(),
            adapter="custom",
            mode=Mode.IMPROVE,
            api_key=None,
            agent_purpose="echo bot",
            storage=storage,
        )
        retuner2.optimize(source="last_n_traces", n=5)
        print("   ✗ Expected RuntimeError but got none")
    except RuntimeError as e:
        print(f"   ✓ Got expected error: {e}")
    except Exception as e:
        print(f"   ? Got unexpected exception type: {type(e).__name__}: {e}")

    print_checklist([
        "Ran `python -m retune dashboard --port 8000 --db ./retune_test_01.db`",
        "Visited http://localhost:8000 → shows 8 traces (5 observe + 3 evaluate)",
        "EVALUATE traces have cost + latency scores in their detail view",
        "No cloud API was contacted (check uvicorn logs if it was running — nothing should appear)",
    ])


if __name__ == "__main__":
    main()
