#!/usr/bin/env bash
# Starts the Python inference server and the Go obfuscation server together.
# Ctrl+C kills both.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INFERENCE_PORT=${INFERENCE_PORT:-8000}
SERVER_PORT=${SERVER_PORT:-8080}
PYTHON="${REPO_ROOT}/.venv/bin/python"

# Start inference backend
cd "$REPO_ROOT/scripts/demo"
"$PYTHON" serve.py --port "$INFERENCE_PORT" &
INFER_PID=$!

# Start Go server
cd "$REPO_ROOT/server"
go run . --addr ":$SERVER_PORT" --inference-url "http://127.0.0.1:$INFERENCE_PORT/infer" &
GO_PID=$!

trap "kill $INFER_PID $GO_PID 2>/dev/null" EXIT INT TERM

echo ""
echo "inference  http://127.0.0.1:$INFERENCE_PORT"
echo "server     http://127.0.0.1:$SERVER_PORT"
echo ""
echo "curl -s -X POST http://127.0.0.1:$SERVER_PORT/obfuscate \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"text\": \"Sean Kearney approved the deployment.\"}' | jq"
echo ""

wait
