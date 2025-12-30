#!/bin/bash
# Script to run the Streamlit web application
# Requires Python 3.10+ (see README for installation instructions)

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Check if Python 3.10 virtual environment exists
if [ -d "venv310" ]; then
    echo "Activating Python 3.10 virtual environment..."
    source venv310/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Warning: No virtual environment found. Please create one with Python 3.10+"
    echo "Run: python3.10 -m venv venv310 && source venv310/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Run Streamlit
echo "Starting Streamlit server on http://localhost:8501"
streamlit run frontend/app.py --server.port 8501 --server.address localhost
