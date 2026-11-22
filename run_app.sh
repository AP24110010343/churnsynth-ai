#!/bin/bash
# One-click startup script for ChurnSynth AI
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "============================================================"
echo " Starting ChurnSynth AI: Tabular Diffusion Model App"
echo "============================================================"

if [ ! -d "venv" ]; then
    echo "Virtual environment not found! Please check setup."
    exit 1
fi

./venv/bin/streamlit run app.py --server.port=8501
