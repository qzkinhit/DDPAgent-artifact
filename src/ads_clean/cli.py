from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .baseline_eval import evaluate_baselines
from .datasets import default_dataset_configs, packaged_suite
from .orchestrator import run_pipeline, run_scenarios, run_suite
from .result_assets import (
    ERROR_RATES,
    copy_result_snapshots,
    find_demandclean_benchmark_root,
    find_uniclean_result_root,
    result_dataset_names,
    scan_asset_catalog,
)
from .summarize import summarize_adsclean_runs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ads-clean", description="Agentic data cleaning pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list-datasets", help="List packaged datasets")

    p_run = sub.add_parser("run", help="Run one end-to-end dataset pipeline")
    p_run.add_argument("--dataset", required=True, help="Dataset name, e.g. beers")
    p_run.add_argument("--detector", choices=["benchmark", "nogt"], default="benchmark")
    p_run.add_argument("--episodes", type=int, default=None)
    p_run.add_argument("--profile", choices=["default", "paper"], default="default")
    p_run.add_argument("--output-root", type=Path, default=None)
    p_run.add_argument("--single-max", type=int, default=10000)
    p_run.add_argument("--scenario", choices=["original", "artificial"], default="original")
    p_run.add_argument("--error-rate", choices=list(ERROR_RATES), default=None)
    p_run.add_argument("--uniclean-result-root", type=Path, default=None)
    p_run.add_argument("--subset-policy", choices=["cluster10k", "direct10k"], default="cluster10k")
    p_run.add_argument("--result-assets", action="store_true", help="Load dataset definitions from UniCleanResult assets")
    _add_demandclean_runtime_args(p_run)
    p_run.add_argument("--quiet", action="store_true")

    p_suite = sub.add_parser("run-suite", help="Run the packaged UniClean-compatible suite")
    p_suite.add_argument("--detector", choices=["benchmark", "nogt"], default="benchmark")
    p_suite.add_argument("--episodes", type=int, default=None)
    p_suite.add_argument("--profile", choices=["default", "paper"], default="default")
    p_suite.add_argument("--output-root", type=Path, default=None)
    _add_demandclean_runtime_args(p_suite)

    p_scenarios = sub.add_parser("run-scenarios", help="Run UniCleanResult original/artificial experiment jobs")
    p_scenarios.add_argument("--scenario", choices=["original", "artificial"], required=True)
    p_scenarios.add_argument("--datasets", nargs="*", default=None)
    p_scenarios.add_argument("--error-rates", nargs="*", choices=list(ERROR_RATES), default=None)
    p_scenarios.add_argument("--detector", choices=["benchmark", "nogt"], default="benchmark")
    p_scenarios.add_argument("--episodes", type=int, default=None)
    p_scenarios.add_argument("--profile", choices=["default", "paper"], default="paper")
    p_scenarios.add_argument("--output-root", type=Path, default=None)
    p_scenarios.add_argument("--single-max", type=int, default=10000)
    p_scenarios.add_argument("--uniclean-result-root", type=Path, default=None)
    p_scenarios.add_argument("--subset-policy", choices=["cluster10k", "direct10k"], default="cluster10k")
    _add_demandclean_runtime_args(p_scenarios)
    p_scenarios.add_argument("--quiet", action="store_true")

    p_baseline = sub.add_parser("eval-baselines", help="Evaluate baseline cleaned CSVs with ADSClean ML tasks")
    p_baseline.add_argument("--scenario", choices=["original", "artificial"], required=True)
    p_baseline.add_argument("--datasets", nargs="*", default=None)
    p_baseline.add_argument("--error-rates", nargs="*", choices=list(ERROR_RATES), default=None)
    p_baseline.add_argument("--output-root", type=Path, default=Path("outputs"))
    p_baseline.add_argument("--uniclean-result-root", type=Path, default=None)
    p_baseline.add_argument("--subset-policy", choices=["cluster10k", "direct10k"], default="cluster10k")

    p_catalog = sub.add_parser("catalog-assets", help="Write a CSV catalog of result assets")
    p_catalog.add_argument("--uniclean-result-root", type=Path, default=None)
    p_catalog.add_argument("--demandclean-benchmark-root", type=Path, default=None)
    p_catalog.add_argument("--output", type=Path, default=Path("result_assets/asset_catalog.csv"))

    p_snapshot = sub.add_parser("prepare-snapshot", help="Copy external result assets into this repo")
    p_snapshot.add_argument("--uniclean-source", type=Path, default=None)
    p_snapshot.add_argument("--demandclean-source", type=Path, default=None)
    p_snapshot.add_argument("--destination", type=Path, default=Path("result_assets"))

    p_summary = sub.add_parser("summarize-runs", help="Merge ADSClean metrics.json files into one CSV")
    p_summary.add_argument("--output-root", type=Path, default=Path("outputs/experiments"))

    args = parser.parse_args(argv)

    if args.command == "list-datasets":
        configs = default_dataset_configs()
        for name in packaged_suite():
            cfg = configs[name]
            print(f"{name}\t{cfg.task_type}\t{cfg.target}\t{cfg.dirty_path}")
        return 0

    if args.command == "run":
        episodes = _episodes(args.profile, args.episodes)
        result = run_pipeline(
            args.dataset,
            output_root=args.output_root,
            detector_mode=args.detector,
            episodes=episodes,
            single_max=args.single_max,
            verbose=not args.quiet,
            scenario=args.scenario,
            error_rate=args.error_rate,
            result_root=args.uniclean_result_root,
            subset_policy=args.subset_policy,
            use_result_assets=args.result_assets,
            rf_estimators=args.rf_estimators,
            rf_max_depth=args.rf_max_depth,
            rf_n_jobs=args.rf_n_jobs,
            reward_eval_interval=args.reward_eval_interval,
            eval_sample_ratio=args.eval_sample_ratio,
            ve_source=args.ve_source,
            delete_policy=args.delete_policy,
            uniclean_scope=args.uniclean_scope,
            detector_expansion=args.detector_expansion,
            max_detector_expansion=args.max_detector_expansion,
            max_detected_errors=args.max_detected_errors,
            base_cv_folds=args.base_cv_folds,
            verifier_policy=args.verifier_policy,
            seed=args.seed,
        )
        print(json.dumps({"run_dir": str(result.run_dir), "cleaned_csv": str(result.cleaned_csv), "metrics": result.metrics}, indent=2, default=str))
        return 0

    if args.command == "run-suite":
        episodes = _episodes(args.profile, args.episodes)
        results = run_suite(
            output_root=args.output_root,
            detector_mode=args.detector,
            episodes=episodes,
            rf_estimators=args.rf_estimators,
            rf_max_depth=args.rf_max_depth,
            rf_n_jobs=args.rf_n_jobs,
            reward_eval_interval=args.reward_eval_interval,
            eval_sample_ratio=args.eval_sample_ratio,
            ve_source=args.ve_source,
            delete_policy=args.delete_policy,
            uniclean_scope=args.uniclean_scope,
            detector_expansion=args.detector_expansion,
            max_detector_expansion=args.max_detector_expansion,
            max_detected_errors=args.max_detected_errors,
            base_cv_folds=args.base_cv_folds,
            verifier_policy=args.verifier_policy,
            seed=args.seed,
        )
        print(json.dumps([{"dataset": r.dataset, "run_dir": str(r.run_dir), "cleaned_csv": str(r.cleaned_csv)} for r in results], indent=2))
        return 0

    if args.command == "run-scenarios":
        episodes = _episodes(args.profile, args.episodes)
        results = run_scenarios(
            output_root=args.output_root,
            detector_mode=args.detector,
            episodes=episodes,
            scenario=args.scenario,
            datasets=args.datasets,
            error_rates=args.error_rates,
            result_root=args.uniclean_result_root,
            subset_policy=args.subset_policy,
            single_max=args.single_max,
            verbose=not args.quiet,
            rf_estimators=args.rf_estimators,
            rf_max_depth=args.rf_max_depth,
            rf_n_jobs=args.rf_n_jobs,
            reward_eval_interval=args.reward_eval_interval,
            eval_sample_ratio=args.eval_sample_ratio,
            ve_source=args.ve_source,
            delete_policy=args.delete_policy,
            uniclean_scope=args.uniclean_scope,
            detector_expansion=args.detector_expansion,
            max_detector_expansion=args.max_detector_expansion,
            max_detected_errors=args.max_detected_errors,
            base_cv_folds=args.base_cv_folds,
            verifier_policy=args.verifier_policy,
            seed=args.seed,
        )
        print(json.dumps([{"dataset": r.dataset, "run_dir": str(r.run_dir), "cleaned_csv": str(r.cleaned_csv)} for r in results], indent=2))
        return 0

    if args.command == "eval-baselines":
        result = evaluate_baselines(
            output_root=args.output_root,
            scenario=args.scenario,
            datasets=args.datasets,
            error_rates=args.error_rates,
            result_root=args.uniclean_result_root,
            subset_policy=args.subset_policy,
        )
        print(json.dumps({"output_dir": str(result.output_dir), "summary_csv": str(result.summary_csv), "rows": len(result.rows)}, indent=2))
        return 0

    if args.command == "catalog-assets":
        df = scan_asset_catalog(args.uniclean_result_root, args.demandclean_benchmark_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(json.dumps({"output": str(args.output), "rows": len(df)}, indent=2))
        return 0

    if args.command == "prepare-snapshot":
        uniclean_source = args.uniclean_source or find_uniclean_result_root(None)
        demandclean_source = args.demandclean_source or find_demandclean_benchmark_root(None)
        dest = copy_result_snapshots(uniclean_source, demandclean_source, args.destination)
        print(json.dumps({"destination": str(dest)}, indent=2))
        return 0

    if args.command == "summarize-runs":
        path = summarize_adsclean_runs(args.output_root)
        print(json.dumps({"summary_csv": str(path)}, indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _episodes(profile: str, explicit: Optional[int]) -> int:
    if explicit is not None:
        return explicit
    return 300 if profile == "paper" else 50


def _add_demandclean_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rf-estimators", type=int, default=25)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    parser.add_argument("--rf-n-jobs", type=int, default=-1)
    parser.add_argument("--reward-eval-interval", type=int, default=0)
    parser.add_argument("--eval-sample-ratio", type=float, default=1.0)
    parser.add_argument("--ve-source", choices=["uniclean", "demandclean"], default="uniclean")
    parser.add_argument("--delete-policy", choices=["execute", "no_op", "uniclean_repair"], default="execute")
    parser.add_argument("--uniclean-scope", choices=["cell", "row"], default="cell")
    parser.add_argument("--detector-expansion", choices=["none", "uniclean_diff"], default="none")
    parser.add_argument("--max-detector-expansion", type=int, default=0)
    parser.add_argument("--max-detected-errors", type=int, default=0)
    parser.add_argument("--base-cv-folds", type=int, default=5)
    parser.add_argument("--verifier-policy", choices=["accept_all", "rollback_no_improve"], default="accept_all")
    parser.add_argument("--seed", type=int, default=42)


if __name__ == "__main__":
    raise SystemExit(main())
