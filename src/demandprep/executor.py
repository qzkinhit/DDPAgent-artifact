from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .policy_runner import ActionPlanResult
from .preprocess import EncodedDataset
from .repair_sources import CleanedValueSource


@dataclass
class ExecutionResult:
    cleaned_df: pd.DataFrame
    cleaned_csv: Path
    repair_source_log: List[Dict[str, object]]
    operation_trace: List[Dict[str, object]]
    fallback_count: int
    policy_override_count: int


ACTION_NAMES = {
    0: "no_op",
    1: "repair_value",
    2: "delete",
    3: "replace_nearby",
}


def execute_final_cleaning(
    encoded: EncodedDataset,
    demand: ActionPlanResult,
    uniclean_source: CleanedValueSource,
    run_dir: Path,
    ve_source: str = "uniclean",
    delete_policy: str = "execute",
    uniclean_scope: str = "cell",
) -> ExecutionResult:
    if ve_source not in {"uniclean", "policy"}:
        raise ValueError("ve_source must be 'uniclean' or 'policy'")
    if delete_policy not in {"execute", "no_op", "uniclean_repair"}:
        raise ValueError("delete_policy must be 'execute', 'no_op', or 'uniclean_repair'")
    if uniclean_scope not in {"cell", "row"}:
        raise ValueError("uniclean_scope must be 'cell' or 'row'")
    raw = encoded.dirty_df.reset_index(drop=True).copy()
    repair_source_log: List[Dict[str, object]] = []
    operation_trace: List[Dict[str, object]] = []
    fallback_count = 0
    policy_override_count = 0
    delete_positions = set()

    for decision in demand.decision_log:
        action = int(decision.get("action", 0))
        row_pos = int(decision["row_idx"])
        col_idx = int(decision["col"])
        if col_idx < 0 or col_idx >= len(encoded.feature_cols) or row_pos >= len(raw):
            continue
        feature_col = encoded.feature_cols[col_idx]
        row_key = _row_key(raw, row_pos, encoded.config.index_col)

        if action == 1:
            _apply_uniclean(
                raw, row_pos, row_key, feature_col, encoded, uniclean_source,
                repair_source_log, operation_trace,
                action_name="repair_value", scope=uniclean_scope,
                strict=True,
            )
        elif action == 3:
            if ve_source == "uniclean":
                applied = _apply_uniclean(
                    raw, row_pos, row_key, feature_col, encoded, uniclean_source,
                    repair_source_log, operation_trace,
                    action_name="replace_nearby", scope=uniclean_scope,
                    strict=False,
                )
                if not applied:
                    fallback_count += 1
                    _apply_decoded_value(raw, row_pos, row_key, feature_col, encoded, decision, repair_source_log, operation_trace)
            else:
                _apply_decoded_value(raw, row_pos, row_key, feature_col, encoded, decision, repair_source_log, operation_trace)
        elif action == 2:
            if delete_policy == "execute":
                if row_pos not in delete_positions:
                    delete_positions.add(row_pos)
                    operation_trace.append(
                        {
                            "operation_id": len(operation_trace),
                            "operation_type": "row_delete",
                            "row_idx": row_key,
                            "row_pos": row_pos,
                            "column": "",
                            "action": "delete",
                            "source": "demandprep_action_policy",
                            "old_value": "",
                            "new_value": "",
                            "changed": True,
                        }
                    )
            elif delete_policy == "uniclean_repair":
                policy_override_count += 1
                _apply_uniclean(
                    raw, row_pos, row_key, feature_col, encoded, uniclean_source,
                    repair_source_log, operation_trace,
                    action_name="delete_as_uniclean_repair", scope=uniclean_scope,
                    strict=True,
                )
            else:
                policy_override_count += 1
                repair_source_log.append(
                    {
                        "row_idx": row_key,
                        "row_pos": row_pos,
                        "column": feature_col,
                        "action": "delete_as_no_op",
                        "source": "governance_delete_policy",
                        "value": raw.loc[row_pos, feature_col],
                    }
                )
    if delete_positions:
        raw = raw.drop(index=sorted(delete_positions)).reset_index(drop=True)

    cleaned_csv = run_dir / "cleaned.csv"
    raw.to_csv(cleaned_csv, index=False)
    return ExecutionResult(raw, cleaned_csv, repair_source_log, operation_trace, fallback_count, policy_override_count)


def _row_key(df: pd.DataFrame, row_pos: int, index_col: str) -> int:
    if index_col in df.columns:
        try:
            return int(float(df.loc[row_pos, index_col]))
        except (TypeError, ValueError):
            pass
    return row_pos


def _apply_uniclean(
    raw: pd.DataFrame,
    row_pos: int,
    row_key: int,
    feature_col: str,
    encoded: EncodedDataset,
    uniclean_source: CleanedValueSource,
    repair_source_log: List[Dict[str, object]],
    operation_trace: List[Dict[str, object]],
    action_name: str,
    scope: str,
    strict: bool,
) -> bool:
    columns = encoded.feature_cols if scope == "row" else [feature_col]
    applied = False
    missing = []
    for col in columns:
        try:
            value = uniclean_source.value_for(row_key, col)
        except KeyError:
            missing.append(col)
            continue
        if col not in raw.columns:
            continue
        before = raw.loc[row_pos, col]
        raw.loc[row_pos, col] = value
        applied = True
        if _norm(before) != _norm(value) or scope == "cell":
            repair_source_log.append(
                {
                    "row_idx": row_key,
                    "row_pos": row_pos,
                    "column": col,
                    "action": action_name,
                    "source": uniclean_source.source_name,
                    "value": value,
                }
            )
            operation_trace.append(
                {
                    "operation_id": len(operation_trace),
                    "operation_type": "cell_update",
                    "row_idx": row_key,
                    "row_pos": row_pos,
                    "column": col,
                    "action": action_name,
                    "source": uniclean_source.source_name,
                    "old_value": before,
                    "new_value": value,
                    "changed": _norm(before) != _norm(value),
                    "source_path": str(uniclean_source.source_path) if uniclean_source.source_path else "",
                }
            )
    if strict and missing:
        raise ValueError(
            f"Missing UniClean execution value for dataset={getattr(encoded.config, 'name', 'unknown')}, "
            f"row_index={row_key}, columns={missing}, source={uniclean_source.source_path}"
        )
    return applied


def _apply_decoded_value(
    raw: pd.DataFrame,
    row_pos: int,
    row_key: int,
    feature_col: str,
    encoded: EncodedDataset,
    decision: Dict[str, object],
    repair_source_log: List[Dict[str, object]],
    operation_trace: List[Dict[str, object]],
) -> None:
    value = encoded.decode_feature_value(feature_col, decision.get("result_value"))
    before = raw.loc[row_pos, feature_col]
    raw.loc[row_pos, feature_col] = value
    repair_source_log.append(
        {
            "row_idx": row_key,
            "row_pos": row_pos,
            "column": feature_col,
            "action": "replace_nearby",
            "source": "demandprep_policy_value_estimator",
            "value": value,
        }
    )
    operation_trace.append(
        {
            "operation_id": len(operation_trace),
            "operation_type": "cell_update",
            "row_idx": row_key,
            "row_pos": row_pos,
            "column": feature_col,
            "action": "replace_nearby",
            "source": "demandprep_policy_value_estimator",
            "old_value": before,
            "new_value": value,
            "changed": _norm(before) != _norm(value),
            "source_path": "",
        }
    )


def _norm(value) -> str:
    if value is None:
        return ""
    return str(value).strip()
