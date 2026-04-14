"""Start a local Postgres in Docker for Retune testing (cross-platform).

Idempotent: if container already exists and running, reports status and exits.
"""
from __future__ import annotations

import subprocess
import sys
import time

CONTAINER = "retune-test-pg"
DB_USER = "retune"
DB_PASS = "devpass"
DB_NAME = "retune"
PORT = 5432


def run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def docker_ps_names(include_stopped: bool = False) -> list[str]:
    args = ["docker", "ps", "--format", "{{.Names}}"]
    if include_stopped:
        args.insert(2, "-a")
    r = run(args, capture=True)
    return r.stdout.strip().splitlines()


def main() -> None:
    try:
        run(["docker", "--version"], capture=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("✗ Docker is not installed or not in PATH.")
        print("  Install Docker Desktop: https://www.docker.com/products/docker-desktop")
        sys.exit(1)

    running = docker_ps_names(include_stopped=False)
    all_names = docker_ps_names(include_stopped=True)

    if CONTAINER in running:
        print(f"✓ {CONTAINER} already running on port {PORT}")
    elif CONTAINER in all_names:
        print(f"→ Starting existing {CONTAINER} container...")
        run(["docker", "start", CONTAINER])
    else:
        print(f"→ Creating new Postgres container {CONTAINER} on port {PORT}...")
        run([
            "docker", "run", "--name", CONTAINER,
            "-e", f"POSTGRES_USER={DB_USER}",
            "-e", f"POSTGRES_PASSWORD={DB_PASS}",
            "-e", f"POSTGRES_DB={DB_NAME}",
            "-p", f"{PORT}:5432",
            "-d", "postgres:15",
        ])

    print("→ Waiting for Postgres to accept connections...")
    for _ in range(30):
        probe = subprocess.run(
            ["docker", "exec", CONTAINER, "pg_isready", "-U", DB_USER],
            capture_output=True,
        )
        if probe.returncode == 0:
            print("✓ Postgres ready")
            print()
            print(f"DATABASE_URL=postgresql://{DB_USER}:{DB_PASS}@localhost:{PORT}/{DB_NAME}")
            return
        time.sleep(1)
    print("✗ Postgres did not become ready within 30s")
    sys.exit(1)


if __name__ == "__main__":
    main()
