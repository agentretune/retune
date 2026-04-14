#!/usr/bin/env bash
# Apply the Alembic initial schema to DATABASE_URL.
set -euo pipefail

# Load .env from testing/setup/.env if present
if [ -f "$(dirname "$0")/.env" ]; then
    export $(grep -v '^#' "$(dirname "$0")/.env" | xargs)
fi

if [ -z "${DATABASE_URL:-}" ]; then
    echo "✗ DATABASE_URL not set. Set it in testing/setup/.env first."
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT/retune-cloud"

echo "→ Running Alembic upgrade head against $DATABASE_URL"
alembic upgrade head

echo ""
echo "✓ Schema applied. Tables created:"
python -c "
import psycopg2, os
conn = psycopg2.connect(os.environ['DATABASE_URL'])
with conn.cursor() as cur:
    cur.execute(\"SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename\")
    for (t,) in cur.fetchall():
        print(f'  - {t}')
conn.close()
"
