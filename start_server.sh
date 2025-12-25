#!/bin/bash
cd "$(dirname "$0")"
pkill -f "streamlit.*app.py" 2>/dev/null
sleep 1
python3 -m streamlit run src/app.py \
    --server.port 8502 \
    --server.address localhost \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.fileWatcherType none
