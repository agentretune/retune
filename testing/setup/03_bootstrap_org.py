"""Create the test organization, user, and API key directly in the DB.

We bypass the `/v1/auth/signup` HTTP route and write to Postgres directly
so this works even when the cloud backend isn't running yet. The cloud
backend's `signup` endpoint does essentially the same thing.

Run after 02_init_db.sh. Prints the API key and credentials for subsequent
scripts. Copy the printed RETUNE_API_KEY into testing/setup/.env.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# Make retune-cloud importable
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "retune-cloud"))

from server.auth import hash_password  # noqa: E402
from server.db import postgres as db  # noqa: E402


def main() -> None:
    email = os.environ.get("TEST_ORG_EMAIL", "test@retune.local")
    password = os.environ.get("TEST_ORG_PASSWORD", "testpass123")
    org_name = os.environ.get("TEST_ORG_NAME", "Retune Test Org")

    if not os.environ.get("DATABASE_URL"):
        print("✗ DATABASE_URL not set. Set it in testing/setup/.env first.")
        sys.exit(1)

    # Skip if already exists
    existing = db.get_user_by_email(email)
    if existing:
        print(f"→ User {email} already exists (org_id={existing['org_id']})")
        print("  Delete via testing/inspect/reset.py to recreate.")
        sys.exit(0)

    # Create org
    slug = org_name.lower().replace(" ", "-")[:50]
    org = db.create_org(name=org_name, slug=slug)
    print(f"✓ Created org {org['id']} ({org['name']})")

    # Create user
    pw_hash = hash_password(password)
    user = db.create_user(
        email=email,
        password_hash=pw_hash,
        name="Test User",
        org_id=org["id"],
        role="admin",
    )
    print(f"✓ Created user {user['id']} ({email})")

    # Create API key
    api_key = db.create_api_key(
        org_id=org["id"],
        name="testing-suite",
        created_by=user["id"],
    )
    print(f"✓ Created API key {api_key['prefix']}...")
    print()
    print("=" * 60)
    print("COPY THESE INTO testing/setup/.env:")
    print("=" * 60)
    print(f"TEST_ORG_EMAIL={email}")
    print(f"TEST_ORG_PASSWORD={password}")
    print(f"RETUNE_API_KEY={api_key['key']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
