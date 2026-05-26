#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest -q tests
python scripts/verify_artifact.py
python scripts/make_ddpagent_figures.py

echo "[reproduce_cached] OK"
