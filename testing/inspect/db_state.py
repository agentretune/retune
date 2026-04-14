"""Print a readable snapshot of the test org's DB state."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / "setup" / ".env")

if not os.environ.get("DATABASE_URL"):
    print("✗ DATABASE_URL not set.")
    sys.exit(1)

import psycopg2  # noqa: E402


def main() -> None:
    email = os.environ.get("TEST_ORG_EMAIL", "test@retune.local")
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()

    cur.execute("SELECT id, name, plan, optimize_runs_used, optimize_runs_limit FROM organizations WHERE slug LIKE %s", ("%retune-test%",))
    org = cur.fetchone()
    if not org:
        print("✗ No test org found. Run testing/setup/03_bootstrap_org.py first.")
        return
    org_id, org_name, plan, used, limit = org
    print()
    print(f"Org: {org_name} ({org_id})")
    print(f"  plan: {plan}")
    print(f"  optimize runs: {used} / {limit}")

    cur.execute(
        "SELECT id, status, source, axes, rewriter_llm, created_at "
        "FROM optimization_runs WHERE org_id = %s ORDER BY created_at DESC LIMIT 10",
        (str(org_id),),
    )
    runs = cur.fetchall()
    print(f"\nRecent runs ({len(runs)}):")
    for r in runs:
        rid, status, source, axes, llm, created = r
        axes_str = ",".join(axes) if axes else "-"
        print(f"  {rid[:20]}... status={status} axes={axes_str} rewriter={llm or '-'} at={created}")

    cur.execute(
        "SELECT run_id, accepted, comment, created_at FROM optimization_feedback "
        "WHERE org_id = %s ORDER BY created_at DESC LIMIT 10",
        (str(org_id),),
    )
    fb = cur.fetchall()
    print(f"\nRecent feedback ({len(fb)}):")
    for r in fb:
        rid, accepted, comment, created = r
        verdict = "✓" if accepted else "✗" if accepted is False else "?"
        c = (comment or "")[:60]
        print(f"  {verdict} run={(rid or '-')[:16]}... {c}")

    cur.execute(
        "SELECT run_id, trace_count, uploaded_at FROM optimization_run_traces "
        "ORDER BY uploaded_at DESC LIMIT 5"
    )
    print(f"\nRecent trace uploads:")
    for r in cur.fetchall():
        rid, n, uploaded = r
        print(f"  {rid[:20]}... traces={n} at={uploaded}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
