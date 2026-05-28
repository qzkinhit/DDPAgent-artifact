from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import random

from demandprep_policy.api import DemandPrepPolicy as ActionAllocator

from .datasets import DatasetConfig, parse_fd_rules
from .preprocess import EncodedDataset


@dataclass
class ActionPlanResult:
    controller: ActionAllocator
    repair_plan: List[Dict[str, Any]]
    decision_log: List[Dict[str, Any]]
    action_counts: Dict[str, int]
    X_current: np.ndarray
    y_current: np.ndarray
    keep_mask: np.ndarray
    detector_mode: str
    raha_used: bool


def run_action_allocation_plan(
    encoded: EncodedDataset,
    run_dir: Path,
    detector_mode: str = "benchmark",
    episodes: int = 50,
    verbose: bool = True,
    rf_estimators: int = 25,
    rf_max_depth: Optional[int] = None,
    rf_n_jobs: int = -1,
    reward_eval_interval: int = 0,
    eval_sample_ratio: float = 1.0,
    semantic_errors: Optional[Sequence[Tuple[int, int]]] = None,
    seed: int = 42,
    max_detected_errors: int = 0,
    base_cv_folds: int = 5,
) -> ActionPlanResult:
    _set_reproducible_seed(seed)
    cfg = encoded.config
    fd_rules = _fd_rules_for_config(cfg)
    use_benchmark_detector = detector_mode == "benchmark" and encoded.work_clean_csv is not None
    disable_raha = detector_mode == "nogt"
    model_kwargs = _model_kwargs(cfg.model_type, rf_estimators, rf_max_depth, rf_n_jobs)

    allocator = ActionAllocator(
        task_type=cfg.task_type,
        model_type=cfg.model_type,
        agent_type="dueling_two_stage",
        detector_mode="auto",
        inference_mode="two_phase",
        training_mode="clean_base",
        n_episodes=episodes,
        repair_lambda=0.03,
        max_repair_ratio=0.3,
        fd_rules=fd_rules,
        column_names=encoded.feature_cols,
        dirty_csv_path=str(encoded.work_dirty_csv),
        clean_csv_path=str(encoded.work_clean_csv) if use_benchmark_detector else None,
        csv_columns=encoded.csv_columns,
        label_col=cfg.target,
        save_path=str(run_dir / "action_allocator"),
        apply_raha_truth=False,
        count_raha_cost=True,
        disable_raha=disable_raha,
        encoding_label_encoders=encoded.feature_encoders,
        encoding_scaler=encoded.scaler,
        encoding_categorical_cols=encoded.categorical_cols,
        encoding_dirty_df=encoded.dirty_df,
        encoding_clean_df=encoded.clean_df,
        model_kwargs=model_kwargs,
        reward_eval_interval=reward_eval_interval,
        eval_sample_ratio=eval_sample_ratio,
        max_detected_errors=max_detected_errors,
        base_cv_folds=base_cv_folds,
        log_interval=max(1, min(50, episodes // 5 if episodes >= 5 else 1)),
    )

    X_clean_for_detector = encoded.X_clean if use_benchmark_detector else None
    y_clean_for_detector = encoded.y_clean if use_benchmark_detector else None
    allocator.fit(
        encoded.X_dirty,
        encoded.y_dirty,
        X_clean=X_clean_for_detector,
        y_clean=y_clean_for_detector,
        semantic_errors=list(semantic_errors) if semantic_errors else None,
        n_episodes=episodes,
        verbose=verbose,
    )
    repair_plan = allocator.plan(
        encoded.X_dirty,
        encoded.y_dirty,
        semantic_errors=list(semantic_errors) if semantic_errors else None,
        verbose=verbose,
    )
    inference = allocator._two_phase_inference
    env = inference._env if inference is not None else None
    if env is None:
        raise RuntimeError("Action allocator did not create a two-phase environment")
    X_current, y_current, keep_mask = env.get_current_data()
    action_counts = env.get_action_counts()
    decision_log = env.get_decision_log()
    raha_used = bool(getattr(allocator.detector, "raha_cost_info", {}).get("raha_total_cost", 0))
    return ActionPlanResult(
        controller=allocator,
        repair_plan=repair_plan,
        decision_log=decision_log,
        action_counts=action_counts,
        X_current=X_current,
        y_current=y_current,
        keep_mask=keep_mask,
        detector_mode=detector_mode,
        raha_used=raha_used,
    )


def _fd_rules_for_config(config: DatasetConfig):
    fd_path = config.dirty_path.parent / "fd_rule.txt"
    return parse_fd_rules(fd_path)


def _model_kwargs(
    model_type: str,
    rf_estimators: int,
    rf_max_depth: Optional[int],
    rf_n_jobs: int,
) -> Optional[Dict[str, Any]]:
    if model_type != "random_forest":
        return None
    kwargs: Dict[str, Any] = {"n_estimators": rf_estimators}
    if rf_max_depth is not None:
        kwargs["max_depth"] = rf_max_depth
    if rf_n_jobs != 0:
        kwargs["n_jobs"] = rf_n_jobs
    return kwargs


def _set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(False)
    except Exception:
        pass
