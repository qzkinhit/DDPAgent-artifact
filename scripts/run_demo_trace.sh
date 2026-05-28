#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/demo_trace_runs}"
EPISODES="${EPISODES:-50}"
RF_ESTIMATORS="${RF_ESTIMATORS:-10}"
BASE_CV_FOLDS="${BASE_CV_FOLDS:-3}"
MAX_DETECTED_ERRORS="${MAX_DETECTED_ERRORS:-0}"
SINGLE_MAX="${SINGLE_MAX:-10000}"
ARTIFICIAL_RATE="${ARTIFICIAL_RATE:-1}"

CLASSIFICATION_DATASETS=(beers flights hospitals rayyan)
REGRESSION_DATASETS=(tax)
CLASSIFICATION_MODELS=(random_forest svm)
REGRESSION_MODELS=(ridge linear random_forest)

run_one() {
  local scenario="$1"
  local dataset="$2"
  local model="$3"
  local rate_arg=()
  local result_assets_arg=()

  if [[ "$scenario" == "artificial" ]]; then
    rate_arg=(--error-rate "$ARTIFICIAL_RATE")
    result_assets_arg=(--result-assets)
  fi

  python -m demandprep.cli run \
    --dataset "$dataset" \
    --scenario "$scenario" \
    "${rate_arg[@]}" \
    "${result_assets_arg[@]}" \
    --output-root "$OUTPUT_ROOT" \
    --profile default \
    --episodes "$EPISODES" \
    --rf-estimators "$RF_ESTIMATORS" \
    --base-cv-folds "$BASE_CV_FOLDS" \
    --max-detected-errors "$MAX_DETECTED_ERRORS" \
    --single-max "$SINGLE_MAX" \
    --model-type-override "$model" \
    --force-uniclean-run \
    --trace-operators \
    --quiet
}

for dataset in "${CLASSIFICATION_DATASETS[@]}"; do
  for model in "${CLASSIFICATION_MODELS[@]}"; do
    run_one original "$dataset" "$model"
    run_one artificial "$dataset" "$model"
  done
done

for dataset in "${REGRESSION_DATASETS[@]}"; do
  for model in "${REGRESSION_MODELS[@]}"; do
    run_one original "$dataset" "$model"
    run_one artificial "$dataset" "$model"
  done
done

python -m demandprep.cli summarize-runs --output-root "$OUTPUT_ROOT"
