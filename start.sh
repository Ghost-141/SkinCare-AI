#!/bin/bash

# Start FastAPI in the background
echo "Starting FastAPI backend..."
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Wait for FastAPI to be ready
echo "Waiting for FastAPI to start (max 180 seconds)..."
for i in {1..180}; do
    if curl -s http://localhost:8000/api/v1/health > /dev/null; then
        echo "FastAPI is ready!"
        break
    fi
    if [ $i -eq 180 ]; then
        echo "FastAPI failed to start within 180 seconds. Exiting."
        exit 1
    fi
    sleep 1
done

# Start Streamlit
echo "Starting Streamlit UI..."
streamlit run ui.py --server.port 8501 --server.address 0.0.0.0
