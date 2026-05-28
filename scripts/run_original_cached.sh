#!/usr/bin/env bash
set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/experiments}"
UNICLEAN_RESULT_ROOT="${UNICLEAN_RESULT_ROOT:-$REPO_ROOT/result_assets/UnicleanResult}"
DETECTOR="${DETECTOR:-benchmark}"
EPISODES="${EPISODES:-300}"
LARGE_EPISODES="${LARGE_EPISODES:-100}"
SUBSET_POLICY="${SUBSET_POLICY:-cluster10k}"
DATASETS="${DATASETS:-hospitals flights beers rayyan tax}"
SINGLE_MAX="${SINGLE_MAX:-10000}"
RF_ESTIMATORS="${RF_ESTIMATORS:-10}"
RF_MAX_DEPTH="${RF_MAX_DEPTH:-}"
RF_N_JOBS="${RF_N_JOBS:--1}"
REWARD_EVAL_INTERVAL="${REWARD_EVAL_INTERVAL:-200}"
LARGE_REWARD_EVAL_INTERVAL="${LARGE_REWARD_EVAL_INTERVAL:-500}"
EVAL_SAMPLE_RATIO="${EVAL_SAMPLE_RATIO:-0.5}"
LARGE_EVAL_SAMPLE_RATIO="${LARGE_EVAL_SAMPLE_RATIO:-0.3}"
BASE_CV_FOLDS="${BASE_CV_FOLDS:-5}"
LARGE_BASE_CV_FOLDS="${LARGE_BASE_CV_FOLDS:-2}"
MAX_DETECTED_ERRORS="${MAX_DETECTED_ERRORS:-0}"
LARGE_MAX_DETECTED_ERRORS="${LARGE_MAX_DETECTED_ERRORS:-500}"
VE_SOURCE="${VE_SOURCE:-uniclean}"
DELETE_POLICY="${DELETE_POLICY:-uniclean_repair}"
UNICLEAN_SCOPE="${UNICLEAN_SCOPE:-row}"
DETECTOR_EXPANSION="${DETECTOR_EXPANSION:-uniclean_diff}"
MAX_DETECTOR_EXPANSION="${MAX_DETECTOR_EXPANSION:-5000}"
LARGE_MAX_DETECTOR_EXPANSION="${LARGE_MAX_DETECTOR_EXPANSION:-1000}"
VERIFIER_POLICY="${VERIFIER_POLICY:-rollback_no_improve}"
SEED="${SEED:-42}"
RESUME="${RESUME:-1}"
QUIET="${QUIET:-1}"

LOG_DIR="$OUTPUT_ROOT/logs/original"
mkdir -p "$LOG_DIR"
FAIL_FILE="$LOG_DIR/failed_jobs.txt"
: > "$FAIL_FILE"

quiet_args=()
if [ "$QUIET" = "1" ]; then
  quiet_args+=(--quiet)
fi

for ds in $DATASETS; do
  job_episodes="$EPISODES"
  job_reward_eval_interval="$REWARD_EVAL_INTERVAL"
  job_eval_sample_ratio="$EVAL_SAMPLE_RATIO"
  job_max_detector_expansion="$MAX_DETECTOR_EXPANSION"
  job_base_cv_folds="$BASE_CV_FOLDS"
  job_max_detected_errors="$MAX_DETECTED_ERRORS"
  case "$ds" in
    tax)
      job_episodes="$LARGE_EPISODES"
      job_reward_eval_interval="$LARGE_REWARD_EVAL_INTERVAL"
      job_eval_sample_ratio="$LARGE_EVAL_SAMPLE_RATIO"
      job_max_detector_expansion="$LARGE_MAX_DETECTOR_EXPANSION"
      job_base_cv_folds="$LARGE_BASE_CV_FOLDS"
      job_max_detected_errors="$LARGE_MAX_DETECTED_ERRORS"
      ;;
  esac
  runtime_args=(--rf-estimators "$RF_ESTIMATORS" --rf-n-jobs "$RF_N_JOBS" --reward-eval-interval "$job_reward_eval_interval" --eval-sample-ratio "$job_eval_sample_ratio" --base-cv-folds "$job_base_cv_folds" --max-detected-errors "$job_max_detected_errors" --ve-source "$VE_SOURCE" --delete-policy "$DELETE_POLICY" --uniclean-scope "$UNICLEAN_SCOPE" --detector-expansion "$DETECTOR_EXPANSION" --max-detector-expansion "$job_max_detector_expansion" --verifier-policy "$VERIFIER_POLICY" --seed "$SEED")
  if [ -n "$RF_MAX_DEPTH" ]; then
    runtime_args+=(--rf-max-depth "$RF_MAX_DEPTH")
  fi
  if [ "$RESUME" = "1" ] && find "$OUTPUT_ROOT/demandprep/original/native/$ds" -name metrics.json -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] original/$ds already has metrics.json"
    continue
  fi
  log="$LOG_DIR/${ds}.log"
  echo "[run] original/$ds -> $log"
  if PYTHONPATH="$REPO_ROOT/src" python -m demandprep.cli run \
      --dataset "$ds" \
      --scenario original \
      --result-assets \
      --uniclean-result-root "$UNICLEAN_RESULT_ROOT" \
      --subset-policy "$SUBSET_POLICY" \
      --detector "$DETECTOR" \
      --episodes "$job_episodes" \
      --single-max "$SINGLE_MAX" \
      --output-root "$OUTPUT_ROOT/demandprep" \
      "${runtime_args[@]}" \
      "${quiet_args[@]}" >"$log" 2>&1; then
    echo "[ok ] original/$ds"
  else
    echo "[fail] original/$ds"
    echo "original,$ds,native,$log" >> "$FAIL_FILE"
  fi
done

if [ -s "$FAIL_FILE" ]; then
  echo "Failures recorded in $FAIL_FILE"
  exit 1
fi
