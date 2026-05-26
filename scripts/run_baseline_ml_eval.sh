#!/usr/bin/env bash
set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/experiments}"
UNICLEAN_RESULT_ROOT="${UNICLEAN_RESULT_ROOT:-$REPO_ROOT/result_assets/UnicleanResult}"
SUBSET_POLICY="${SUBSET_POLICY:-cluster10k}"

LOG_DIR="$OUTPUT_ROOT/logs/baselines"
mkdir -p "$LOG_DIR"
FAIL_FILE="$LOG_DIR/failed_jobs.txt"
: > "$FAIL_FILE"

run_eval() {
  local scenario="$1"
  local log="$LOG_DIR/${scenario}.log"
  echo "[run] baseline ML $scenario -> $log"
  if PYTHONPATH="$REPO_ROOT/src" python -m ads_clean.cli eval-baselines \
      --scenario "$scenario" \
      --uniclean-result-root "$UNICLEAN_RESULT_ROOT" \
      --subset-policy "$SUBSET_POLICY" \
      --output-root "$OUTPUT_ROOT" >"$log" 2>&1; then
    echo "[ok ] baseline ML $scenario"
  else
    echo "[fail] baseline ML $scenario"
    echo "$scenario,$log" >> "$FAIL_FILE"
  fi
}

SCENARIOS="${SCENARIOS:-original artificial}"
for scenario in $SCENARIOS; do
  run_eval "$scenario"
done

if [ -s "$FAIL_FILE" ]; then
  echo "Failures recorded in $FAIL_FILE"
  exit 1
fi
