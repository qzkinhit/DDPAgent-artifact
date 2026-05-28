from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd

from .datasets import load_dataset_config, packaged_suite
from .executor import execute_final_cleaning
from .metrics import evaluate_cell_repair, evaluate_downstream
from .paths import default_output_root, project_root
from .policy_runner import run_action_allocation_plan
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
    force_uniclean_run: bool = False,
    trace_operators: bool = False,
    model_type_override: Optional[str] = None,
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
    if model_type_override:
        cfg = replace(cfg, model_type=model_type_override)
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
        force_uniclean_run=force_uniclean_run,
        trace_operators=trace_operators,
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
    demand = run_action_allocation_plan(
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

    _mark_operation_acceptance(execution.operation_trace, verifier_selected)

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
        "force_uniclean_run": force_uniclean_run,
        "trace_operators": trace_operators,
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
    pd.DataFrame(demand.decision_log).to_csv(run_dir / "action_trace.csv", index=False)
    pd.DataFrame(demand.repair_plan).to_csv(run_dir / "repair_plan.csv", index=False)
    pd.DataFrame(execution.repair_source_log).to_csv(run_dir / "repair_source_log.csv", index=False)
    pd.DataFrame(execution.operation_trace).to_csv(run_dir / "operation_trace.csv", index=False)
    _copy_uniclean_operator_traces(run_dir, uniclean.trace)
    _write_model_trace(run_dir / "model_trace.csv", metrics)
    _write_workflow_trace(run_dir / "workflow_trace.json", metrics, uniclean.trace, demand.decision_log, execution.operation_trace)
    _write_json(
        run_dir / "run_config.json",
        {
            "dataset": cfg.name,
            "scenario": cfg.scenario,
            "error_rate": cfg.error_rate,
            "subset_source": cfg.subset_source,
            "dirty_path": _artifact_path(cfg.dirty_path),
            "clean_path": _artifact_path(cfg.clean_path),
            "cached_uniclean_path": _artifact_path(cfg.cached_uniclean_path),
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
            "force_uniclean_run": force_uniclean_run,
            "trace_operators": trace_operators,
            "model_type_override": model_type_override,
            "expanded_semantic_errors": int(len(expanded_semantic_errors)),
            "seed": seed,
            "single_max": single_max,
            "notes": cfg.notes,
        },
    )

    return RunResult(cfg.name, run_dir, execution.cleaned_csv, metrics)


def _artifact_path(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(project_root().resolve()))
    except ValueError:
        return str(path)


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
    force_uniclean_run: bool = False,
    trace_operators: bool = False,
    model_type_override: Optional[str] = None,
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
            force_uniclean_run=force_uniclean_run,
            trace_operators=trace_operators,
            model_type_override=model_type_override,
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
    force_uniclean_run: bool = False,
    trace_operators: bool = False,
    model_type_override: Optional[str] = None,
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
                force_uniclean_run=force_uniclean_run,
                trace_operators=trace_operators,
                model_type_override=model_type_override,
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
    force_uniclean_run: bool,
    trace_operators: bool,
):
    if not force_uniclean_run and not trace_operators and cfg.cached_uniclean_path and Path(cfg.cached_uniclean_path).exists():
        return load_cached_uniclean(cfg, dirty_df, run_dir)
    if not allow_uniclean_run:
        raise FileNotFoundError(f"Cached UniClean file not found: {cfg.cached_uniclean_path}")
    uniclean = run_uniclean(
        cfg,
        dirty_csv,
        run_dir,
        single_max=single_max,
        verbose=False,
        trace_operators=trace_operators,
    )
    if cfg.cached_uniclean_path and not force_uniclean_run and not trace_operators:
        cache_path = Path(cfg.cached_uniclean_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(uniclean.cleaned_csv, cache_path)
        uniclean.trace["cached_after_run"] = str(cache_path)
    return uniclean


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def _mark_operation_acceptance(operation_trace, verifier_selected: str) -> None:
    accepted = verifier_selected == "candidate"
    for row in operation_trace:
        row["accepted_by_verifier"] = accepted
        row["verifier_selected"] = verifier_selected


def _copy_uniclean_operator_traces(run_dir: Path, trace: Dict[str, object]) -> None:
    for name in ("operator_trace.json", "operator_trace.csv", "operation_rule_trace.csv", "operator_weight_trace.csv"):
        key = name.replace(".", "_")
        source = trace.get(key)
        if not source:
            continue
        source_path = Path(str(source))
        if source_path.exists():
            dest = run_dir / name
            if source_path.resolve() != dest.resolve():
                shutil.copyfile(source_path, dest)


def _write_model_trace(path: Path, metrics: Dict[str, object]) -> None:
    fields = [
        "dataset",
        "scenario",
        "error_rate",
        "task_type",
        "model_type",
        "downstream_before",
        "downstream_after",
        "downstream_delta",
        "downstream_fixed_before",
        "downstream_fixed_after",
        "downstream_fixed_delta",
        "verifier_policy",
        "verifier_selected",
    ]
    row = {field: metrics.get(field, "") for field in fields}
    pd.DataFrame([row]).to_csv(path, index=False)


def _write_workflow_trace(
    path: Path,
    metrics: Dict[str, object],
    uniclean_trace: Dict[str, object],
    decision_log,
    operation_trace,
) -> None:
    action_names = {0: "no_op", 1: "repair_value", 2: "delete", 3: "replace_nearby"}
    action_counts = {name: 0 for name in action_names.values()}
    for row in decision_log:
        name = action_names.get(int(row.get("action", 0)), "unknown")
        action_counts[name] = action_counts.get(name, 0) + 1

    operator_files = {
        "operator_trace": path.parent / "operator_trace.csv",
        "operation_rule_trace": path.parent / "operation_rule_trace.csv",
        "operator_weight_trace": path.parent / "operator_weight_trace.csv",
    }
    capabilities = {
        "action_trace": (path.parent / "action_trace.csv").exists(),
        "operation_trace": (path.parent / "operation_trace.csv").exists(),
        "model_trace": (path.parent / "model_trace.csv").exists(),
        "runtime_operator_trace": any(p.exists() and p.stat().st_size > 0 for p in operator_files.values()),
        "cached_uniclean": bool(uniclean_trace.get("cached", False)),
    }
    payload = {
        "trace_version": 1,
        "dataset": metrics.get("dataset"),
        "scenario": metrics.get("scenario"),
        "error_rate": metrics.get("error_rate"),
        "task_type": metrics.get("task_type"),
        "model_type": metrics.get("model_type"),
        "verifier_selected": metrics.get("verifier_selected"),
        "capabilities": capabilities,
        "action_counts": action_counts,
        "operation_count": len(operation_trace),
        "accepted_operation_count": sum(1 for row in operation_trace if row.get("accepted_by_verifier")),
        "uniclean_candidate_repairs": metrics.get("uniclean_candidate_repairs"),
        "trace_files": {
            "action_trace": str(path.parent / "action_trace.csv"),
            "operation_trace": str(path.parent / "operation_trace.csv"),
            "model_trace": str(path.parent / "model_trace.csv"),
            **{key: str(value) for key, value in operator_files.items() if value.exists()},
        },
    }
    _write_json(path, payload)


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
