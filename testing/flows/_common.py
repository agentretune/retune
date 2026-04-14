"""Shared utilities for flow scripts."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def load_env() -> None:
    """Load testing/setup/.env and add repo paths to sys.path."""
    repo_root = Path(__file__).parent.parent.parent
    env_path = repo_root / "testing" / "setup" / ".env"
    if not env_path.exists():
        print(f"✗ Missing {env_path} — copy env.example and fill it in first.")
        sys.exit(1)
    load_dotenv(env_path)

    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root / "testing"))
    sys.path.insert(0, str(repo_root / "retune-cloud"))


def require_env(*names: str) -> None:
    """Exit if any required env var is missing."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        print(f"✗ Missing env vars: {', '.join(missing)}")
        print("  Set them in testing/setup/.env and re-run.")
        sys.exit(1)


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_checklist(items: list[str]) -> None:
    print()
    print("VERIFY:")
    for item in items:
        print(f"  [ ] {item}")
    print()
