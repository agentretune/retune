"""CLI entry point: python -m retune dashboard"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="retune",
        description="Retune -- self-improving agent optimization framework",
    )
    sub = parser.add_subparsers(dest="command")

    # Dashboard command
    dash = sub.add_parser("dashboard", help="Launch the retune dashboard")
    dash.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    dash.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    dash.add_argument("--db", default="retune.db", help="SQLite database path")
    dash.add_argument("--no-open", action="store_true", help="Don't auto-open browser")

    # Version command
    sub.add_parser("version", help="Show retune version")

    args = parser.parse_args()

    if args.command == "version":
        from retune._version import __version__
        print(f"retune {__version__}")

    elif args.command == "dashboard":
        _run_dashboard(args)

    else:
        parser.print_help()


def _run_dashboard(args: argparse.Namespace) -> None:
    """Start the dashboard server."""
    import os
    os.environ["RETUNE_STORAGE_PATH"] = args.db
    os.environ["RETUNE_HOST"] = args.host
    os.environ["RETUNE_PORT"] = str(args.port)

    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required for the dashboard.")
        print("Install with: pip install retune[server]")
        sys.exit(1)

    url = f"http://localhost:{args.port}"
    print(f"Starting retune dashboard at {url}")
    print(f"Database: {args.db}")
    print()

    if not args.no_open:
        import threading
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(
        "retune.dashboard.app:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
