#!/usr/bin/env bash
# Launch the CardioRisk BN demonstration website locally.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.11 || command -v python3)}"

if [[ ! -d .venv ]]; then
  echo "Creating virtual environment with ${PYTHON_BIN}"
  "${PYTHON_BIN}" -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt

echo "Starting demo at http://localhost:8501"
exec streamlit run app/streamlit_app.py --server.headless=false
