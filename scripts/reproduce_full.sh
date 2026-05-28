#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/rerun_$(date +%Y%m%d_%H%M%S)}"
export UNICLEAN_RESULT_ROOT="${UNICLEAN_RESULT_ROOT:-$ROOT/result_assets/UnicleanResult}"

bash scripts/run_original_cached.sh
bash scripts/run_artificial_cached.sh
bash scripts/run_baseline_ml_eval.sh
python -m demandprep.cli summarize-runs --output-root "$OUTPUT_ROOT/demandprep"

echo "[reproduce_full] results written to $OUTPUT_ROOT"
