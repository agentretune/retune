"""Delete the test org and all its data — start from clean slate."""
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
    resp = input("⚠️  This deletes all test-org data permanently. Continue? [y/N] ")
    if resp.lower() != "y":
        print("Aborted.")
        return

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()

    # Find the test org by slug pattern
    cur.execute("SELECT id FROM organizations WHERE slug LIKE %s", ("%retune-test%",))
    rows = cur.fetchall()
    if not rows:
        print("No test org found.")
        return

    for (org_id,) in rows:
        cur.execute("DELETE FROM organizations WHERE id = %s", (org_id,))
        print(f"✓ Deleted org {org_id} (cascaded: users, traces, runs, reports, feedback)")

    conn.commit()
    cur.close()
    conn.close()

    print("\nNow re-run testing/setup/03_bootstrap_org.py to create a fresh org.")


if __name__ == "__main__":
    main()
