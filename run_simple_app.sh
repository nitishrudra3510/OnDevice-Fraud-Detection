#!/bin/bash

echo "🚀 Starting Simplified Streamlit App (No Mutex Errors)..."

# Kill any existing processes
pkill -f "streamlit run" 2>/dev/null || true
sleep 2

# Go to project root and activate environment
cd /Users/nitishkumar/Documents/GitHub/Multi-Agent
source .venv/bin/activate

# Start the simplified app that works without errors
streamlit run OnDevice-Fraud-Detection/streamlit_app_simple.py \
    --server.port 8508 \
    --server.enableCORS false \
    --server.enableXsrfProtection false

echo "✅ Streamlit app is now running at http://localhost:8508"
echo "🌐 Open your browser and go to: http://localhost:8508"
