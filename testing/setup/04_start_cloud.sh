#!/usr/bin/env bash
# Start the cloud backend. Leave this running in its own terminal.
set -euo pipefail

if [ -f "$(dirname "$0")/.env" ]; then
    export $(grep -v '^#' "$(dirname "$0")/.env" | xargs)
fi

if [ -z "${DATABASE_URL:-}" ] || [ -z "${RETUNE_JWT_SECRET:-}" ]; then
    echo "✗ DATABASE_URL and RETUNE_JWT_SECRET must be set in testing/setup/.env"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT/retune-cloud"

echo "→ Starting retune-cloud on http://localhost:8001"
echo "  Swagger: http://localhost:8001/docs"
echo "  Ctrl+C to stop"
echo ""
exec uvicorn server.app:app --host 0.0.0.0 --port 8001 --reload
