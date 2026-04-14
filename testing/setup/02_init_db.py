"""Apply Alembic migrations to DATABASE_URL (cross-platform)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        print(f"✗ Missing {env_path} — copy env.example and fill it in.")
        sys.exit(1)
    load_dotenv(env_path)

    if not os.environ.get("DATABASE_URL"):
        print("✗ DATABASE_URL not set in testing/setup/.env")
        sys.exit(1)

    repo_root = Path(__file__).parent.parent.parent
    cloud_dir = repo_root / "retune-cloud"

    print(f"→ Running `alembic upgrade head` in {cloud_dir}")
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=cloud_dir,
        env={**os.environ},
    )
    if result.returncode != 0:
        print("✗ Alembic upgrade failed")
        sys.exit(result.returncode)

    print("\n✓ Schema applied. Tables:")
    import psycopg2
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname='public' ORDER BY tablename"
            )
            for (t,) in cur.fetchall():
                print(f"  - {t}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
