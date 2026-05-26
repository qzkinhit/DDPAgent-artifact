from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd

from .datasets import load_dataset_config, packaged_suite
from .demandclean_runner import run_demandclean_plan
from .executor import execute_final_cleaning
from .metrics import evaluate_cell_repair, evaluate_downstream
from .paths import default_output_root
from .preprocess import prepare_dataset
from .result_assets import ERROR_RATES, build_result_dataset_config, result_dataset_names
from .uniclean_runner import load_cached_uniclean, run_uniclean


@dataclass
class RunResult:
    dataset: str
    run_dir: Path
    cleaned_csv: Path
    metrics: Dict[str, object]


def run_pipeline(
    dataset: str,
    output_root: Optional[Path] = None,
    detector_mode: str = "benchmark",
    episodes: int = 50,
    single_max: int = 10000,
    verbose: bool = True,
    scenario: str = "original",
    error_rate: Optional[str] = None,
    result_root: Optional[Path] = None,
    subset_policy: str = "cluster10k",
    allow_uniclean_run: bool = True,
    use_result_assets: bool = False,
    rf_estimators: int = 25,
    rf_max_depth: Optional[int] = None,
    rf_n_jobs: int = -1,
    reward_eval_interval: int = 0,
    eval_sample_ratio: float = 1.0,
    ve_source: str = "uniclean",
    delete_policy: str = "execute",
    uniclean_scope: str = "cell",
    detector_expansion: str = "none",
    max_detector_expansion: int = 0,
    max_detected_errors: int = 0,
    base_cv_folds: int = 5,
    verifier_policy: str = "accept_all",
    seed: int = 42,
) -> RunResult:
    if detector_mode not in {"benchmark", "nogt"}:
        raise ValueError("detector_mode must be 'benchmark' or 'nogt'")
    if detector_expansion not in {"none", "uniclean_diff"}:
        raise ValueError("detector_expansion must be 'none' or 'uniclean_diff'")
    if verifier_policy not in {"accept_all", "rollback_no_improve"}:
        raise ValueError("verifier_policy must be 'accept_all' or 'rollback_no_improve'")

    start = time.perf_counter()
    if use_result_assets or result_root is not None or scenario != "original" or error_rate is not None:
        cfg = build_result_dataset_config(
            dataset,
            scenario=scenario,
            error_rate=error_rate,
            result_root=result_root,
            subset_policy=subset_policy,
        )
    else:
        cfg = load_dataset_config(dataset)
    out_root = Path(output_root) if output_root else default_output_root()
    scenario_name = cfg.scenario if cfg.scenario else scenario
    rate_name = cfg.error_rate or "native"
    run_dir = out_root / scenario_name / rate_name / cfg.name / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    encoded = prepare_dataset(cfg, run_dir / "work")
    uniclean = _load_or_run_uniclean(
        cfg,
        encoded.dirty_df,
        encoded.work_dirty_csv,
        run_dir,
        single_max=single_max,
        allow_uniclean_run=allow_uniclean_run,
    )
    expanded_semantic_errors = (
        _uniclean_diff_semantic_errors(
            encoded,
            uniclean.value_source,
            max_positions=max_detector_expansion,
        )
        if detector_expansion == "uniclean_diff"
        else []
    )
    demand = run_demandclean_plan(
        encoded,
        run_dir=run_dir,
        detector_mode=detector_mode,
        episodes=episodes,
        verbose=verbose,
        rf_estimators=rf_estimators,
        rf_max_depth=rf_max_depth,
        rf_n_jobs=rf_n_jobs,
        reward_eval_interval=reward_eval_interval,
        eval_sample_ratio=eval_sample_ratio,
        semantic_errors=expanded_semantic_errors,
        seed=seed,
        max_detected_errors=max_detected_errors,
        base_cv_folds=base_cv_folds,
    )
    execution = execute_final_cleaning(
        encoded,
        demand,
        uniclean.value_source,
        run_dir,
        ve_source=ve_source,
        delete_policy=delete_policy,
        uniclean_scope=uniclean_scope,
    )

    downstream_metrics = evaluate_downstream(encoded, execution.cleaned_df)
    cell_metrics = evaluate_cell_repair(encoded, execution.cleaned_df)
    verifier_selected = "candidate"
    verifier_candidate: Dict[str, object] = {}
    if verifier_policy == "rollback_no_improve":
        before = downstream_metrics.get("downstream_fixed_before")
        after = downstream_metrics.get("downstream_fixed_after")
        if before is not None and after is not None and after < before:
            candidate_csv = run_dir / "candidate_cleaned.csv"
            shutil.copyfile(execution.cleaned_csv, candidate_csv)
            verifier_candidate = {
                "verifier_candidate_cleaned_csv": str(candidate_csv),
                "verifier_candidate_fixed_before": before,
                "verifier_candidate_fixed_after": after,
                "verifier_candidate_fixed_delta": downstream_metrics.get("downstream_fixed_delta"),
                "verifier_candidate_repair_f1": cell_metrics.get("repair_f1"),
            }
            rollback_df = encoded.dirty_df.reset_index(drop=True).copy()
            rollback_df.to_csv(execution.cleaned_csv, index=False)
            execution.cleaned_df = rollback_df
            downstream_metrics = evaluate_downstream(encoded, execution.cleaned_df)
            cell_metrics = evaluate_cell_repair(encoded, execution.cleaned_df)
            verifier_selected = "no_op_rollback"

    metrics: Dict[str, object] = {
        "dataset": cfg.name,
        "scenario": cfg.scenario,
        "error_rate": cfg.error_rate,
        "subset_source": cfg.subset_source,
        "task_type": cfg.task_type,
        "model_type": cfg.model_type,
        "target": cfg.target,
        "detector_mode": detector_mode,
        "benchmark_assisted_detection": detector_mode == "benchmark",
        "raha_used": demand.raha_used,
        "episodes": episodes,
        "rf_estimators": rf_estimators,
        "rf_max_depth": rf_max_depth,
        "rf_n_jobs": rf_n_jobs,
        "reward_eval_interval": reward_eval_interval,
        "eval_sample_ratio": eval_sample_ratio,
        "ve_source": ve_source,
        "delete_policy": delete_policy,
        "uniclean_scope": uniclean_scope,
        "detector_expansion": detector_expansion,
        "max_detector_expansion": max_detector_expansion,
        "max_detected_errors": max_detected_errors,
        "base_cv_folds": base_cv_folds,
        "verifier_policy": verifier_policy,
        "verifier_selected": verifier_selected,
        "expanded_semantic_errors": int(len(expanded_semantic_errors)),
        "seed": seed,
        "rows": int(len(encoded.dirty_df)),
        "dropped_rows": int(encoded.dropped_rows),
        "feature_count": int(len(encoded.feature_cols)),
        "categorical_feature_count": int(len(encoded.categorical_cols)),
        "action_counts": demand.action_counts,
        "repair_plan_size": int(len(demand.repair_plan)),
        "uniclean_candidate_repairs": int(len(uniclean.candidate_repairs)),
        "uniclean_cached": bool(uniclean.trace.get("cached", False)),
        "uniclean_cleaned_csv": str(uniclean.cleaned_csv),
        "repair_fallback_count": int(execution.fallback_count),
        "policy_override_count": int(execution.policy_override_count),
        "runtime_seconds": time.perf_counter() - start,
    }
    metrics.update(verifier_candidate)
    metrics.update(downstream_metrics)
    metrics.update(cell_metrics)

    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "uniclean_trace.json", uniclean.trace)
    pd.DataFrame(demand.decision_log).to_csv(run_dir / "decision_log.csv", index=False)
    pd.DataFrame(demand.repair_plan).to_csv(run_dir / "repair_plan.csv", index=False)
    pd.DataFrame(execution.repair_source_log).to_csv(run_dir / "repair_source_log.csv", index=False)
    _write_json(
        run_dir / "run_config.json",
        {
            "dataset": cfg.name,
            "scenario": cfg.scenario,
            "error_rate": cfg.error_rate,
            "subset_source": cfg.subset_source,
            "dirty_path": str(cfg.dirty_path),
            "clean_path": str(cfg.clean_path) if cfg.clean_path else None,
            "cached_uniclean_path": str(cfg.cached_uniclean_path) if cfg.cached_uniclean_path else None,
            "detector_mode": detector_mode,
            "episodes": episodes,
            "rf_estimators": rf_estimators,
            "rf_max_depth": rf_max_depth,
            "rf_n_jobs": rf_n_jobs,
            "reward_eval_interval": reward_eval_interval,
            "eval_sample_ratio": eval_sample_ratio,
            "ve_source": ve_source,
            "delete_policy": delete_policy,
            "uniclean_scope": uniclean_scope,
            "detector_expansion": detector_expansion,
            "max_detector_expansion": max_detector_expansion,
            "max_detected_errors": max_detected_errors,
            "base_cv_folds": base_cv_folds,
            "verifier_policy": verifier_policy,
            "expanded_semantic_errors": int(len(expanded_semantic_errors)),
            "seed": seed,
            "single_max": single_max,
            "notes": cfg.notes,
        },
    )

    return RunResult(cfg.name, run_dir, execution.cleaned_csv, metrics)


def run_suite(
    output_root: Optional[Path] = None,
    detector_mode: str = "benchmark",
    episodes: int = 50,
    rf_estimators: int = 25,
    rf_max_depth: Optional[int] = None,
    rf_n_jobs: int = -1,
    reward_eval_interval: int = 0,
    eval_sample_ratio: float = 1.0,
    ve_source: str = "uniclean",
    delete_policy: str = "execute",
    uniclean_scope: str = "cell",
    detector_expansion: str = "none",
    max_detector_expansion: int = 0,
    max_detected_errors: int = 0,
    base_cv_folds: int = 5,
    verifier_policy: str = "accept_all",
    seed: int = 42,
):
    return [
        run_pipeline(
            name,
            output_root=output_root,
            detector_mode=detector_mode,
            episodes=episodes,
            rf_estimators=rf_estimators,
            rf_max_depth=rf_max_depth,
            rf_n_jobs=rf_n_jobs,
            reward_eval_interval=reward_eval_interval,
            eval_sample_ratio=eval_sample_ratio,
            ve_source=ve_source,
            delete_policy=delete_policy,
            uniclean_scope=uniclean_scope,
            detector_expansion=detector_expansion,
            max_detector_expansion=max_detector_expansion,
            max_detected_errors=max_detected_errors,
            base_cv_folds=base_cv_folds,
            verifier_policy=verifier_policy,
            seed=seed,
        )
        for name in packaged_suite()
    ]


def run_scenarios(
    output_root: Optional[Path] = None,
    detector_mode: str = "benchmark",
    episodes: int = 50,
    scenario: str = "original",
    datasets: Optional[Sequence[str]] = None,
    error_rates: Optional[Sequence[str]] = None,
    result_root: Optional[Path] = None,
    subset_policy: str = "cluster10k",
    single_max: int = 10000,
    verbose: bool = True,
    rf_estimators: int = 25,
    rf_max_depth: Optional[int] = None,
    rf_n_jobs: int = -1,
    reward_eval_interval: int = 0,
    eval_sample_ratio: float = 1.0,
    ve_source: str = "uniclean",
    delete_policy: str = "execute",
    uniclean_scope: str = "cell",
    detector_expansion: str = "none",
    max_detector_expansion: int = 0,
    max_detected_errors: int = 0,
    base_cv_folds: int = 5,
    verifier_policy: str = "accept_all",
    seed: int = 42,
) -> Sequence[RunResult]:
    names = tuple(datasets) if datasets else tuple(result_dataset_names())
    if scenario == "original":
        jobs = [(name, None) for name in names]
    elif scenario == "artificial":
        rates = tuple(error_rates) if error_rates else tuple(ERROR_RATES)
        jobs = [(name, rate) for name in names for rate in rates]
    else:
        raise ValueError("scenario must be 'original' or 'artificial'")

    results = []
    for name, rate in jobs:
        results.append(
            run_pipeline(
                name,
                output_root=output_root,
                detector_mode=detector_mode,
                episodes=episodes,
                single_max=single_max,
                verbose=verbose,
                scenario=scenario,
                error_rate=rate,
                result_root=result_root,
                subset_policy=subset_policy,
                use_result_assets=True,
                rf_estimators=rf_estimators,
                rf_max_depth=rf_max_depth,
                rf_n_jobs=rf_n_jobs,
                reward_eval_interval=reward_eval_interval,
                eval_sample_ratio=eval_sample_ratio,
                ve_source=ve_source,
                delete_policy=delete_policy,
                uniclean_scope=uniclean_scope,
                detector_expansion=detector_expansion,
                max_detector_expansion=max_detector_expansion,
                max_detected_errors=max_detected_errors,
                base_cv_folds=base_cv_folds,
                verifier_policy=verifier_policy,
                seed=seed,
            )
        )
    return results


def _load_or_run_uniclean(
    cfg,
    dirty_df,
    dirty_csv: Path,
    run_dir: Path,
    single_max: int,
    allow_uniclean_run: bool,
):
    if cfg.cached_uniclean_path and Path(cfg.cached_uniclean_path).exists():
        return load_cached_uniclean(cfg, dirty_df, run_dir)
    if not allow_uniclean_run:
        raise FileNotFoundError(f"Cached UniClean file not found: {cfg.cached_uniclean_path}")
    uniclean = run_uniclean(cfg, dirty_csv, run_dir, single_max=single_max, verbose=False)
    if cfg.cached_uniclean_path:
        cache_path = Path(cfg.cached_uniclean_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(uniclean.cleaned_csv, cache_path)
        uniclean.trace["cached_after_run"] = str(cache_path)
    return uniclean


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def _uniclean_diff_semantic_errors(
    encoded,
    uniclean_source,
    max_positions: int = 0,
) -> Sequence[Tuple[int, int]]:
    dirty = encoded.dirty_df.reset_index(drop=True)
    cleaned = uniclean_source.project_to(dirty).reset_index(drop=True)
    n = min(len(dirty), len(cleaned))
    positions = []
    for row_pos in range(n):
        for col_idx, col in enumerate(encoded.feature_cols):
            if col not in dirty.columns or col not in cleaned.columns:
                continue
            if _norm_cell(dirty.loc[row_pos, col]) != _norm_cell(cleaned.loc[row_pos, col]):
                positions.append((row_pos, col_idx))
    if max_positions > 0 and len(positions) > max_positions:
        step = (len(positions) - 1) / max(max_positions - 1, 1)
        keep = [positions[round(i * step)] for i in range(max_positions)]
        return keep
    return positions


def _norm_cell(value) -> str:
    if value is None:
        return ""
    return str(value).strip()
