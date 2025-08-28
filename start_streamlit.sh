#!/bin/bash

echo "🚀 Starting Streamlit App..."

# Kill any existing processes
pkill -f "streamlit run" 2>/dev/null || true
sleep 2

# Go to project root and activate environment
cd /Users/nitishkumar/Documents/GitHub/Multi-Agent
source .venv/bin/activate

# Start Streamlit with minimal config to avoid crashes
streamlit run OnDevice-Fraud-Detection/streamlit_app.py \
    --server.port 8507 \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.headless true \
    --logger.level error

echo "✅ Streamlit should now be running at http://localhost:8507"
