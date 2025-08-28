#!/bin/bash
cd /Users/nitishkumar/Documents/GitHub/Multi-Agent && source .venv/bin/activate && streamlit run OnDevice-Fraud-Detection/streamlit_app.py --server.port 8507 --server.enableCORS false --server.enableXsrfProtection false --server.headless true
