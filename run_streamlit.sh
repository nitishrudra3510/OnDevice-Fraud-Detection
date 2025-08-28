#!/bin/bash

# Kill any existing Streamlit processes
pkill -f "streamlit run" 2>/dev/null || true

# Activate virtual environment
source ../.venv/bin/activate

# Run Streamlit with fixed configuration
streamlit run streamlit_app.py \
    --server.port 8505 \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.headless false

echo "Streamlit app started at http://localhost:8505"
