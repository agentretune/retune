"""Hit every major cloud route and print status codes."""
from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / "setup" / ".env")

BASE = os.environ.get("RETUNE_CLOUD_BASE_URL", "http://localhost:8001")
KEY = os.environ.get("RETUNE_API_KEY", "")


def ping(method: str, path: str, expected: set[int]) -> None:
    url = f"{BASE}{path}"
    req = urllib.request.Request(url, method=method)
    if KEY:
        req.add_header("Authorization", f"Bearer {KEY}")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            code = resp.status
    except urllib.error.HTTPError as e:
        code = e.code
    except Exception as e:
        print(f"  {method:6s} {path:50s} → ERROR: {e}")
        return
    mark = "✓" if code in expected else "✗"
    print(f"  {mark} {method:6s} {path:50s} → {code}")


def main() -> None:
    print(f"Cloud: {BASE}")
    if not KEY:
        print("  (no RETUNE_API_KEY — some 401s are expected)")

    print("\nPublic:")
    ping("GET", "/docs", {200})
    ping("GET", "/openapi.json", {200})

    print("\nAuth:")
    ping("GET", "/api/v1/auth/verify", {200, 401})

    print("\nOptimize (needs auth):")
    ping("GET", "/api/v1/optimize/runs?limit=5", {200, 401})
    ping("POST", "/api/v1/optimize/preauthorize", {422, 401})  # missing body → 422

    print("\nBilling:")
    ping("GET", "/api/v1/billing/usage", {200, 401})
    ping("GET", "/api/v1/billing/plans", {200})


if __name__ == "__main__":
    main()
