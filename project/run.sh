#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(dirname "$DIR")"

echo "=========================================="
echo "    Starting Minimal RAG System Server"
echo "=========================================="
echo "Directory: $DIR"
echo "Open browser at: http://localhost:8000"
echo "=========================================="

cd "$DIR"
exec "$ROOT_DIR/.venv/bin/python" -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

