# DDPAgent Artifact

[中文说明](README.zh-CN.md)

This repository contains the code, datasets, cached cleaning outputs, paper artifacts, and interactive demo used by the ADS 2026 workshop paper:

**DDPAgent for Demand-Driven Data Preparation via Agentic Action Allocation and Operator-Grounded Execution**

The artifact is scoped to the paper. It includes only the datasets, baselines, cached cleaning outputs, and experiment summaries used by the paper and the demo.

## Core Idea

DDPAgent treats data preparation as a data-governance agent rather than a single fixed cleaning pipeline. Given a downstream task, model type, budget, candidate actions, and available data operators, the system first trains an action-allocation policy without paying human ground-truth cost. At inference time, the policy chooses what to do for each detected error or data object, such as no-op, repair, delete, or value replacement. Each action then dispatches to its own operator space. In the cleaning instantiation used in the paper, repair and replacement actions use a multi-signal cleaner orchestration layer that generates concrete repair rules and execution traces.

This design is intentionally extensible. The action space is not limited to cleaning. Future actions can include data augmentation, data selection, model-tuning decisions, or other data-preparation operations. Likewise, each action can maintain its own operator library and orchestration strategy.

## Truthfulness And Provenance

The repository does not contain simulated workflow traces. The default demo view reads real cached run artifacts for efficiency. When a runtime operator trace is unavailable because a run used a cached cleaned table, the UI explicitly says so and only displays the real traces that exist.

The Streamlit demo also provides a real-run button. It invokes:

```bash
python -m ads_clean.cli run ... --force-uniclean-run --trace-operators
```

That command reruns UniClean and DDPAgent instead of fabricating operator coverage, rule counts, weights, or data operations.

## Contents

- `src/ads_clean`: DDPAgent orchestration, dataset loading, action allocation execution, trace writing, baseline evaluation, and summarization.
- `src/demandclean`: vendored RL action-allocation implementation used by the controller.
- `src/SampleScrubber`, `src/AnalyticsCache`, `src/CoreSetSample`: vendored operator-execution substrate used by the cleaning executor.
- `data/uniclean`: packaged native-error tables for Beers, Flights, Hospitals, Rayyan, and Tax-10K.
- `result_assets/UnicleanResult`: cached native and injected tables, full-operator outputs, and fixed-cleaner outputs used by the paper.
- `outputs/experiments_20260519_final` and `outputs/experiments_20260520_hospital_measurecode`: paper experiment summaries and run artifacts.
- `outputs/demo_trace_runs`: real trace-enabled run artifacts used by the interactive demo.
- `paper`: paper source, compiled PDF, bibliography, and generated figures.
- `streamlit_app.py`: interactive workflow explorer.

## Scope

Included datasets are Beers, Flights, Hospitals, Rayyan, and Tax-10K.

Fixed-cleaner baselines included in the artifact are Baran, BigDansing, Holistic, HoloClean, and Horizon. Diagnostic controls used by the paper include No-op, FullOps, OracleDel, and GTRepair. OracleDel and GTRepair require clean-reference information and are not deployable fixed cleaning baselines.

The cached fixed-cleaner tables are used to reproduce the paper's downstream ML evaluation. This artifact does not rerun Baran, BigDansing, Holistic, HoloClean, or Horizon from scratch.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `torch` or `pyspark` installation is slow on your platform, install official wheels for your Python version first, then rerun the requirements command.

## Cached Reproduction

The fastest verification path uses included cached tables and summaries.

```bash
bash scripts/reproduce_cached.sh
```

This runs unit tests, validates artifact scope, checks reported result summaries, and regenerates paper figures from included experiment tables.

## Full Experiment Rerun

To rerun DDPAgent experiments against cached cleaning outputs, use:

```bash
bash scripts/reproduce_full.sh
```

This trains the action allocator for the five native-error datasets and all 40 injected settings, then evaluates fixed-cleaner baselines with the same downstream tasks. Runtime depends on CPU resources. Tax uses the clustered 10K subset included in the artifact.

## Trace-Enabled Demo Rerun

To generate real workflow traces for the UI, use:

```bash
bash scripts/run_demo_trace.sh
```

This script runs native and injected settings for the paper datasets with multiple downstream models and enables runtime operator tracing. It writes results to `outputs/demo_trace_runs`.

A smaller single run can be launched manually:

```bash
PYTHONPATH=src python -m ads_clean.cli run \
  --dataset beers \
  --scenario original \
  --output-root outputs/demo_trace_runs \
  --episodes 50 \
  --model-type-override random_forest \
  --force-uniclean-run \
  --trace-operators
```

## Interactive Agent Console

Start the UI with:

```bash
bash scripts/run_streamlit_demo.sh
```

Then open the URL printed by Streamlit, usually:

```text
http://127.0.0.1:8501
```

The UI is organized as an agent console rather than a static result browser. It first asks the user to configure the data-governance task:

1. choose a dataset
2. choose native or injected-error setting
3. choose candidate downstream models and an active model
4. choose cached artifact inspection or real execution

The UI defaults to real cached artifacts. Select **Run real pipeline** in the execution step to launch a new real run. The global language button in the sidebar switches between English and Chinese.

The workflow explorer shows:

- task and downstream model configuration
- action allocation over detected errors
- runtime operator orchestration when trace files exist
- generated repair rules and feedback weights when collected
- verifier-labeled data operations with before and after values
- downstream utility before and after preparation

## Result Provenance

Main paper summaries:

- `outputs/experiments_20260519_final/adsclean/adsclean_summary.csv`
- `outputs/experiments_20260519_final/baseline_eval/original/baseline_ml_summary.csv`
- `outputs/experiments_20260519_final/baseline_eval/artificial/baseline_ml_summary.csv`
- `outputs/experiments_20260520_hospital_measurecode/adsclean/adsclean_summary.csv`
- `outputs/experiments_20260520_hospital_measurecode/baseline_eval/original/baseline_ml_summary.csv`
- `outputs/experiments_20260520_hospital_measurecode/baseline_eval/artificial/baseline_ml_summary.csv`

Trace-enabled runs write:

- `action_trace.csv`
- `operation_trace.csv`
- `operator_trace.csv`
- `operation_rule_trace.csv`
- `operator_weight_trace.csv`
- `model_trace.csv`
- `workflow_trace.json`

Cached runs may not include runtime operator traces because cached cleaned tables do not preserve the original operator-level rule provenance.

## License

Code is released under the MIT License. Dataset and cached baseline outputs are included for research reproduction of the paper experiments. Original upstream dataset licenses may apply.
