# DDPAgent Artifact

This repository contains the code, datasets, cached cleaning outputs, and experimental summaries used by the ADS 2026 workshop paper:

**DDPAgent for Demand-Driven Data Preparation via Agentic Action Allocation and Operator-Grounded Execution**

The artifact is intentionally scoped to the paper. It includes only the datasets and fixed-cleaner baselines reported in the paper.

## Contents

- `src/ads_clean`: the DDPAgent orchestration code, including dataset loading, action allocation, execution, baseline evaluation, and summarization.
- `src/demandclean`: the vendored RL action-allocation implementation used by the controller.
- `src/SampleScrubber`, `src/AnalyticsCache`, `src/CoreSetSample`: the vendored operator-execution substrate used by the cleaning executor.
- `data/uniclean`: compact packaged native-error tables for Beers, Flights, Hospitals, Rayyan, and Tax.
- `result_assets/UnicleanResult`: cached native and injected tables, full-operator outputs, and fixed-cleaner outputs used by the paper.
- `outputs`: real experiment summaries and run artifacts used for the paper tables and figures.
- `paper`: the paper source, compiled PDF, bibliography, and generated figures.

## Scope

The included datasets are Beers, Flights, Hospitals, Rayyan, and Tax-10K.

The included fixed-cleaner baselines are Baran, BigDansing, Holistic, HoloClean, and Horizon. The repository also contains diagnostic outputs used in the paper, including No-op, FullOps, OracleDel, and GTRepair. OracleDel and GTRepair require clean-reference information and are not deployable baselines.

The cached fixed-cleaner CSVs are used to reproduce the paper's downstream ML evaluation. This artifact does not rerun those external cleaning systems from scratch.

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `torch` or `pyspark` installation is slow on your platform, install them from the official wheels for your Python version first, then rerun the requirements command.

## One-command cached reproduction

The fastest verification path uses the included cached tables and paper summaries.

```bash
bash scripts/reproduce_cached.sh
```

This command runs unit tests, validates that the artifact only contains the paper datasets and baselines, checks the reported result summaries, and regenerates the paper figures from the included experiment tables.

## Full rerun

To rerun the DDPAgent experiments against the cached cleaning outputs, use:

```bash
bash scripts/reproduce_full.sh
```

The full rerun trains the action allocator for all five native-error datasets and all 40 injected settings, then evaluates the fixed-cleaner baselines with the same downstream tasks. Runtime depends on CPU resources. Tax-10K uses the clustered 10K subset included in the artifact.

## Result provenance

The main paper summaries are:

- `outputs/experiments_20260519_final/adsclean/adsclean_summary.csv`
- `outputs/experiments_20260519_final/baseline_eval/original/baseline_ml_summary.csv`
- `outputs/experiments_20260519_final/baseline_eval/artificial/baseline_ml_summary.csv`
- `outputs/experiments_20260520_hospital_measurecode/adsclean/adsclean_summary.csv`
- `outputs/experiments_20260520_hospital_measurecode/baseline_eval/original/baseline_ml_summary.csv`
- `outputs/experiments_20260520_hospital_measurecode/baseline_eval/artificial/baseline_ml_summary.csv`

Per-run artifacts include `metrics.json`, `decision_log.csv`, `repair_plan.csv`, `repair_source_log.csv`, `cleaned.csv`, and `uniclean_trace.json` where available.

## License

Code is released under the MIT License. Dataset and cached baseline outputs are included for research reproduction of the paper experiments. Original upstream dataset licenses may apply.
