"""Start the cloud backend. Leave this running in its own terminal.

Cross-platform — invokes uvicorn via python -m so it works on Windows without
activating virtualenvs first (assuming uvicorn is installed in the current env).
"""
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

    required = ("DATABASE_URL", "RETUNE_JWT_SECRET")
    missing = [n for n in required if not os.environ.get(n)]
    if missing:
        print(f"✗ Missing env vars: {', '.join(missing)}")
        sys.exit(1)

    repo_root = Path(__file__).parent.parent.parent
    cloud_dir = repo_root / "retune-cloud"

    print("→ Starting retune-cloud on http://localhost:8001")
    print("  Swagger: http://localhost:8001/docs")
    print("  Ctrl+C to stop")
    print()
    # Use exec semantics: this process is replaced by uvicorn.
    try:
        subprocess.run(
            [
                sys.executable, "-m", "uvicorn",
                "server.app:app",
                "--host", "0.0.0.0",
                "--port", "8001",
                "--reload",
            ],
            cwd=cloud_dir,
            env={**os.environ},
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
