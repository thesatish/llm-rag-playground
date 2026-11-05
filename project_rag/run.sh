#!/usr/bin/env bash
set -e

# Resolve script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Activate virtualenv if available
if [ -f "${PARENT_DIR}/.venv/bin/activate" ]; then
    source "${PARENT_DIR}/.venv/bin/activate"
elif [ -f "${SCRIPT_DIR}/.venv/bin/activate" ]; then
    source "${SCRIPT_DIR}/.venv/bin/activate"
fi

MODE="${1:-server}"

if [ "$MODE" == "cli" ]; then
    echo "Starting Advanced RAG Interactive CLI..."
    python3 "${SCRIPT_DIR}/cli.py"
elif [ "$MODE" == "test" ]; then
    echo "Running Advanced RAG Test Suite..."
    python3 "${SCRIPT_DIR}/tests/test_rag_pipeline.py"
else
    echo "=========================================================="
    echo "Starting Advanced RAG Platform Web Server..."
    echo "Open your browser at: http://localhost:8001"
    echo "API Swagger Docs at:  http://localhost:8001/docs"
    echo "=========================================================="
    python3 -m uvicorn app:app --app-dir "${SCRIPT_DIR}" --host 0.0.0.0 --port 8001 --reload
fi

