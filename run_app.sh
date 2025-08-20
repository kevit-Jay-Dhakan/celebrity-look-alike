#!/bin/bash
set -e

# Ensure repository root is on the Python path
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Launch FastAPI backend in the background
python apps/platform/src/app.py &

# Start the Streamlit web interface
python apps/platform_web/src/app.py

