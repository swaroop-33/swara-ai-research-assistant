#!/bin/sh

# ==============================================================================
# start.sh
# ==============================================================================
# Runner script to orchestrate FastAPI backend and Streamlit frontend.
# Works in Hugging Face Spaces (UID 1000 sandboxed environment).
# Ensure this file is saved with LF line endings.

echo "=== [1/4] Starting FastAPI ==="
# Launch FastAPI backend on localhost:8000 in the background
uvicorn main:app --host 127.0.0.1 --port 8000 --log-level info &
FASTAPI_PID=$!

echo "=== [2/4] Waiting for API ==="
# Setup timeout config (60 seconds max wait)
TIMEOUT=60
ELAPSED=0
HEALTH_URL="http://127.0.0.1:8000/api/v1/health"

while [ $ELAPSED -lt $TIMEOUT ]; do
    # 1. Verify that the background FastAPI process hasn't already terminated
    if ! kill -0 $FASTAPI_PID 2>/dev/null; then
        echo "Error: FastAPI backend failed to start or crashed."
        exit 1
    fi

    # 2. Test HTTP connection to health check endpoint
    curl -s -f "$HEALTH_URL" >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "=== [3/4] API Ready ==="
        break
    fi

    # 3. Increment counters
    echo "Waiting for API... ($ELAPSED seconds elapsed)"
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

# Check if we broke out of loop due to timeout
if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "Error: Timeout waiting for FastAPI backend to start after ${TIMEOUT} seconds."
    kill $FASTAPI_PID 2>/dev/null || true
    exit 1
fi

echo "=== [4/4] Starting Streamlit ==="
# Execute Streamlit in the foreground. Using 'exec' makes Streamlit the primary 
# container process (PID 1), ensuring proper signal handling and shutdown.
exec streamlit run frontend/app.py --server.port 7860 --server.address 0.0.0.0
