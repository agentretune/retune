#!/usr/bin/env bash
# Start a local Postgres in Docker for Retune testing.
# Idempotent: if container already exists and running, prints status and exits.
set -euo pipefail

CONTAINER="retune-test-pg"
DB_USER="retune"
DB_PASS="devpass"
DB_NAME="retune"
PORT=5432

if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "✓ $CONTAINER already running on port $PORT"
elif docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "→ Starting existing $CONTAINER container..."
    docker start "$CONTAINER"
else
    echo "→ Creating new Postgres container $CONTAINER on port $PORT..."
    docker run --name "$CONTAINER" \
        -e POSTGRES_USER="$DB_USER" \
        -e POSTGRES_PASSWORD="$DB_PASS" \
        -e POSTGRES_DB="$DB_NAME" \
        -p "$PORT:5432" \
        -d postgres:15
fi

# Wait for it to be ready
echo "→ Waiting for Postgres to accept connections..."
for i in {1..30}; do
    if docker exec "$CONTAINER" pg_isready -U "$DB_USER" >/dev/null 2>&1; then
        echo "✓ Postgres ready"
        echo ""
        echo "DATABASE_URL=postgresql://$DB_USER:$DB_PASS@localhost:$PORT/$DB_NAME"
        exit 0
    fi
    sleep 1
done
echo "✗ Postgres did not become ready within 30s"
exit 1
