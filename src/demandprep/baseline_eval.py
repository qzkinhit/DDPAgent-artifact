from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .metrics import evaluate_cell_repair, evaluate_downstream
from .preprocess import EncodedDataset, prepare_dataset
from .repair_sources import CleanedValueSource, normalize_index_column
from .result_assets import (
    ERROR_RATES,
    baseline_cleaned_candidates,
    build_result_dataset_config,
    result_dataset_names,
)


BASELINE_SYSTEMS: Sequence[str] = ("baran", "bigdansing", "holistic", "holoclean", "horizon")


@dataclass
class BaselineEvalResult:
    output_dir: Path
    summary_csv: Path
    rows: List[Dict[str, object]]


def evaluate_baselines(
    output_root: Path,
    scenario: str = "original",
    datasets: Optional[Sequence[str]] = None,
    error_rates: Optional[Sequence[str]] = None,
    result_root: Optional[Path] = None,
    subset_policy: str = "cluster10k",
) -> BaselineEvalResult:
    output_dir = Path(output_root) / "baseline_eval" / scenario
    output_dir.mkdir(parents=True, exist_ok=True)
    names = tuple(datasets) if datasets else tuple(result_dataset_names())
    rates = (None,) if scenario == "original" else tuple(error_rates) if error_rates else tuple(ERROR_RATES)

    rows: List[Dict[str, object]] = []
    for name in names:
        for rate in rates:
            cfg = build_result_dataset_config(
                name,
                scenario=scenario,
                error_rate=rate,
                result_root=result_root,
                subset_policy=subset_policy,
            )
            scenario_label = rate or "native"
            work_dir = output_dir / cfg.name / scenario_label / "work"
            encoded = prepare_dataset(cfg, work_dir)
            rows.extend(_evaluate_reference_strategies(encoded, output_dir, scenario_label))
            rows.extend(_evaluate_external_baselines(encoded, output_dir, scenario_label, result_root))

    summary = pd.DataFrame(rows)
    summary_csv = output_dir / "baseline_ml_summary.csv"
    summary.to_csv(summary_csv, index=False)
    with (output_dir / "baseline_ml_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False, default=str)
    return BaselineEvalResult(output_dir=output_dir, summary_csv=summary_csv, rows=rows)


def _evaluate_reference_strategies(
    encoded: EncodedDataset,
    output_dir: Path,
    scenario_label: str,
) -> List[Dict[str, object]]:
    rows = []
    rows.append(_score_strategy(encoded, encoded.dirty_df, "no_op", "reference", "main", None))
    if encoded.clean_df is not None:
        rows.append(_score_strategy(encoded, encoded.clean_df, "oracle_full_repair", "oracle", "main", None))
        rows.append(
            _score_strategy(
                encoded,
                _delete_true_error_rows(encoded),
                "delete_true_error_rows",
                "oracle",
                "main",
                None,
            )
        )
    if encoded.config.cached_uniclean_path and Path(encoded.config.cached_uniclean_path).exists():
        source = CleanedValueSource.from_csv(
            Path(encoded.config.cached_uniclean_path),
            index_col=encoded.config.index_col,
            source_name="uniclean_full",
        )
        rows.append(
            _score_strategy(
                encoded,
                _ensure_schema(source.project_to(encoded.dirty_df), encoded.dirty_df),
                "uniclean_full",
                "uniclean_result",
                "main",
                encoded.config.cached_uniclean_path,
            )
        )
    return rows


def _evaluate_external_baselines(
    encoded: EncodedDataset,
    output_dir: Path,
    scenario_label: str,
    result_root: Optional[Path],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for system in BASELINE_SYSTEMS:
        candidates = baseline_cleaned_candidates(
            encoded.config.name,
            system,
            encoded.config.scenario,
            encoded.config.error_rate,
            result_root=result_root,
        )
        if not candidates:
            rows.append(_missing_row(encoded, system, scenario_label))
            continue
        for path in candidates:
            try:
                cleaned = _load_baseline_cleaned(path, encoded.dirty_df)
                source = CleanedValueSource.from_df(cleaned, path, index_col=encoded.config.index_col, source_name=system)
                projected = _ensure_schema(source.project_to(encoded.dirty_df), encoded.dirty_df)
                coverage = _index_coverage(encoded.dirty_df, cleaned, encoded.config.index_col)
                group = "main" if coverage >= 0.95 else "appendix_partial_subset"
                rows.append(_score_strategy(encoded, projected, system, "baseline_cleaned_csv", group, path, coverage))
            except Exception as exc:
                row = _missing_row(encoded, system, scenario_label)
                row.update({"status": "error", "error": str(exc), "source_path": str(path)})
                rows.append(row)
    return rows


def _score_strategy(
    encoded: EncodedDataset,
    cleaned_df: pd.DataFrame,
    strategy: str,
    provenance: str,
    comparison_group: str,
    source_path,
    coverage: Optional[float] = 1.0,
) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    metrics.update(evaluate_downstream(encoded, cleaned_df))
    metrics.update(evaluate_cell_repair(encoded, cleaned_df))
    return {
        "dataset": encoded.config.name,
        "scenario": encoded.config.scenario,
        "error_rate": encoded.config.error_rate or "native",
        "subset_source": encoded.config.subset_source,
        "strategy": strategy,
        "provenance": provenance,
        "comparison_group": comparison_group,
        "source_path": str(source_path) if source_path else "",
        "row_count": int(len(cleaned_df)),
        "coverage": coverage,
        "status": "ok",
        **metrics,
    }


def _missing_row(encoded: EncodedDataset, system: str, scenario_label: str) -> Dict[str, object]:
    return {
        "dataset": encoded.config.name,
        "scenario": encoded.config.scenario,
        "error_rate": encoded.config.error_rate or "native",
        "subset_source": encoded.config.subset_source,
        "strategy": system,
        "provenance": "baseline_cleaned_csv",
        "comparison_group": "missing",
        "source_path": "",
        "row_count": 0,
        "coverage": 0.0,
        "status": "missing",
    }


def _delete_true_error_rows(encoded: EncodedDataset) -> pd.DataFrame:
    dirty = normalize_index_column(encoded.dirty_df, encoded.config.index_col).reset_index(drop=True)
    clean = normalize_index_column(encoded.clean_df, encoded.config.index_col).reset_index(drop=True)
    if encoded.config.index_col in dirty.columns and encoded.config.index_col in clean.columns:
        dirty = dirty.set_index(encoded.config.index_col, drop=False)
        clean = clean.set_index(encoded.config.index_col, drop=False)
        common = dirty.index.intersection(clean.index)
        dirty = dirty.loc[common].reset_index(drop=True)
        clean = clean.loc[common].reset_index(drop=True)
    cols = [c for c in encoded.feature_cols if c in dirty.columns and c in clean.columns]
    error_mask = pd.Series(False, index=dirty.index)
    for col in cols:
        error_mask |= dirty[col].map(_norm_cell) != clean[col].map(_norm_cell)
    return dirty.loc[~error_mask].reset_index(drop=True)


def _load_baseline_cleaned(path: Path, reference_df: pd.DataFrame) -> pd.DataFrame:
    cleaned = pd.read_csv(path, dtype=str, keep_default_na=False)
    cleaned = normalize_index_column(cleaned, "index")
    reference_cols = list(reference_df.columns)

    if list(cleaned.columns) != reference_cols and len(cleaned.columns) == len(reference_cols):
        cleaned.columns = reference_cols
        cleaned = normalize_index_column(cleaned, "index")
    elif "index" not in cleaned.columns:
        recovered_index = _recover_index_from_key(cleaned, reference_df)
        if recovered_index is not None:
            cleaned.insert(0, "index", recovered_index)
        elif len(cleaned) == len(reference_df):
            cleaned.insert(0, "index", reference_df["index"].astype(str).tolist())

    return _ensure_schema(cleaned, reference_df)


def _recover_index_from_key(cleaned: pd.DataFrame, reference_df: pd.DataFrame) -> Optional[List[str]]:
    for key in ("id", "ProviderNumber", "provider_number", "tno"):
        if key not in cleaned.columns or key not in reference_df.columns:
            continue
        ref = reference_df[["index", key]].copy()
        if ref[key].duplicated().any() or cleaned[key].duplicated().any():
            continue
        index_by_key = dict(zip(ref[key].astype(str), ref["index"].astype(str)))
        recovered = cleaned[key].astype(str).map(index_by_key)
        if recovered.notna().mean() >= 0.95:
            return recovered.fillna("").astype(str).tolist()
    return None


def _ensure_schema(cleaned: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_index_column(cleaned, "index")
    reference = normalize_index_column(reference_df, "index")
    if "index" not in out.columns and "index" in reference.columns and len(out) == len(reference):
        out.insert(0, "index", reference["index"].astype(str).tolist())
    if "index" in out.columns and "index" in reference.columns:
        ref_indexed = reference.set_index("index", drop=False)
        out_indexed = out.set_index("index", drop=False)
        for col in reference.columns:
            if col not in out_indexed.columns:
                out_indexed[col] = out_indexed.index.map(lambda idx: ref_indexed.loc[idx, col] if idx in ref_indexed.index else "")
        out = out_indexed[[c for c in reference.columns if c in out_indexed.columns]].reset_index(drop=True)
    else:
        for col in reference.columns:
            if col not in out.columns:
                out[col] = reference[col].iloc[: len(out)].reset_index(drop=True)
        out = out[[c for c in reference.columns if c in out.columns]]
    return out


def _index_coverage(reference_df: pd.DataFrame, cleaned_df: pd.DataFrame, index_col: str) -> float:
    ref = normalize_index_column(reference_df, index_col)
    cleaned = normalize_index_column(cleaned_df, index_col)
    if index_col not in ref.columns or index_col not in cleaned.columns or ref.empty:
        return 0.0
    ref_set = set(ref[index_col].astype(str))
    cleaned_set = set(cleaned[index_col].astype(str))
    return len(ref_set & cleaned_set) / len(ref_set)


def _norm_cell(value) -> str:
    if value is None:
        return ""
    return str(value).strip()
