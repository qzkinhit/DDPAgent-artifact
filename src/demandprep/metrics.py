from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from .datasets import MISSING_TOKENS
from .preprocess import EncodedDataset


def evaluate_downstream(encoded: EncodedDataset, cleaned_df: pd.DataFrame) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if len(cleaned_df) < 10:
        return metrics

    base_df = encoded.dirty_df.reset_index(drop=True)
    aligned_df, retained_mask = _align_to_base(encoded, cleaned_df)
    keep_positions = np.where(retained_mask)[0]
    result_df = aligned_df.iloc[keep_positions].reset_index(drop=True)
    X_dirty = encoded.X_dirty[keep_positions]
    y = encoded.y_dirty[keep_positions]

    X_cleaned = encoded.encode_features(result_df)
    y_cleaned = _encode_target_frame(encoded, result_df)[: len(X_cleaned)]

    n = min(len(X_dirty), len(X_cleaned), len(y_cleaned))
    X_dirty = _fill_nan(X_dirty[:n])
    X_cleaned = _fill_nan(X_cleaned[:n])
    y = y[:n]
    y_cleaned = y_cleaned[:n]

    valid = ~np.isnan(y_cleaned)
    if encoded.config.task_type == "regression":
        valid &= ~np.isnan(y)
    if valid.sum() < 10:
        return metrics
    X_dirty, X_cleaned, y = X_dirty[valid], X_cleaned[valid], y[valid]

    try:
        idx = np.arange(len(y))
        train_idx, test_idx = train_test_split(idx, test_size=0.3, random_state=42)
        if encoded.config.task_type == "regression":
            before = _regression_score(X_dirty[train_idx], y[train_idx], X_dirty[test_idx], y[test_idx], encoded.config.model_type)
            after = _regression_score(X_cleaned[train_idx], y[train_idx], X_cleaned[test_idx], y[test_idx], encoded.config.model_type)
            metrics.update({"downstream_before": before, "downstream_after": after, "downstream_delta": after - before})
        else:
            before = _classification_score(X_dirty[train_idx], y[train_idx], X_dirty[test_idx], y[test_idx], encoded.config.model_type)
            after = _classification_score(X_cleaned[train_idx], y[train_idx], X_cleaned[test_idx], y[test_idx], encoded.config.model_type)
            metrics.update({"downstream_before": before, "downstream_after": after, "downstream_delta": after - before})
    except Exception as exc:
        metrics["downstream_error"] = str(exc)
    metrics.update(_evaluate_fixed_split_downstream(encoded, aligned_df, retained_mask))
    return metrics


def _align_to_base(encoded: EncodedDataset, cleaned_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    base_df = encoded.dirty_df.reset_index(drop=True)
    result_df = cleaned_df.reset_index(drop=True)
    aligned = base_df.copy()
    retained = np.zeros(len(base_df), dtype=bool)
    index_col = encoded.config.index_col

    if index_col in base_df.columns and index_col in result_df.columns:
        result = result_df.copy()
        result[index_col] = result[index_col].astype(str)
        result = result.drop_duplicates(index_col, keep="first").set_index(index_col, drop=False)
        for pos, idx in enumerate(base_df[index_col].astype(str)):
            if idx not in result.index:
                continue
            row = result.loc[idx]
            for col in base_df.columns:
                if col in row.index:
                    aligned.at[pos, col] = row[col]
            retained[pos] = True
    else:
        n_pos = min(len(base_df), len(result_df))
        if n_pos > 0:
            for col in base_df.columns:
                if col in result_df.columns:
                    aligned.loc[: n_pos - 1, col] = result_df[col].iloc[:n_pos].to_numpy()
            retained[:n_pos] = True
    return aligned, retained


def _evaluate_fixed_split_downstream(
    encoded: EncodedDataset,
    aligned_cleaned_df: pd.DataFrame,
    retained_mask: np.ndarray,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if len(encoded.y_dirty) < 10 or retained_mask.sum() < 10:
        return metrics
    try:
        X_dirty = _fill_nan(encoded.X_dirty)
        X_cleaned = _fill_nan(encoded.encode_features(aligned_cleaned_df))
        X_test = _fill_nan(encoded.X_clean if encoded.X_clean is not None else encoded.X_dirty)
        y_train = encoded.y_dirty
        y_train_after = _encode_target_frame(encoded, aligned_cleaned_df)
        y_test = encoded.y_clean if encoded.y_clean is not None else encoded.y_dirty

        n = min(len(X_dirty), len(X_cleaned), len(X_test), len(y_train), len(y_train_after), len(y_test), len(retained_mask))
        X_dirty = X_dirty[:n]
        X_cleaned = X_cleaned[:n]
        X_test = X_test[:n]
        y_train = y_train[:n]
        y_train_after = y_train_after[:n]
        y_test = y_test[:n]
        retained_mask = retained_mask[:n]

        valid = ~np.isnan(y_train) & ~np.isnan(y_train_after) & ~np.isnan(y_test)
        valid_idx = np.where(valid)[0]
        if len(valid_idx) < 10:
            return metrics
        train_idx, test_idx = train_test_split(valid_idx, test_size=0.3, random_state=42)
        after_train_idx = train_idx[retained_mask[train_idx]]
        if len(after_train_idx) < 10 or len(test_idx) < 10:
            return metrics

        if encoded.config.task_type == "regression":
            before = _regression_score(
                X_dirty[train_idx], y_train[train_idx],
                X_test[test_idx], y_test[test_idx],
                encoded.config.model_type,
            )
            after = _regression_score(
                X_cleaned[after_train_idx], y_train_after[after_train_idx],
                X_test[test_idx], y_test[test_idx],
                encoded.config.model_type,
            )
        else:
            before = _classification_score(
                X_dirty[train_idx], y_train[train_idx],
                X_test[test_idx], y_test[test_idx],
                encoded.config.model_type,
            )
            after = _classification_score(
                X_cleaned[after_train_idx], y_train_after[after_train_idx],
                X_test[test_idx], y_test[test_idx],
                encoded.config.model_type,
            )
        metrics.update({
            "downstream_fixed_before": before,
            "downstream_fixed_after": after,
            "downstream_fixed_delta": after - before,
            "downstream_fixed_train_rows_after": int(len(after_train_idx)),
            "downstream_fixed_test_rows": int(len(test_idx)),
        })
    except Exception as exc:
        metrics["downstream_fixed_error"] = str(exc)
    return metrics


def _encode_target_frame(encoded: EncodedDataset, df: pd.DataFrame) -> np.ndarray:
    if encoded.config.target not in df.columns:
        return encoded.y_dirty.copy()
    values = df[encoded.config.target].map(_clean_target_value)
    if encoded.config.task_type == "regression":
        return pd.to_numeric(values, errors="coerce").astype(float).to_numpy()
    if encoded.label_encoder is None:
        return encoded.y_dirty.copy()
    lookup = {klass: idx for idx, klass in enumerate(encoded.label_encoder.classes_)}
    return values.astype(str).map(lambda value: lookup.get(value, np.nan)).astype(float).to_numpy()


def _clean_target_value(value):
    if value is None:
        return np.nan
    text = str(value).strip()
    if text in MISSING_TOKENS:
        return np.nan
    return text


def evaluate_cell_repair(encoded: EncodedDataset, cleaned_df: pd.DataFrame) -> Dict[str, float]:
    if encoded.clean_df is None:
        return {}
    index_col = encoded.config.index_col
    dirty = encoded.dirty_df.reset_index(drop=True)
    clean = encoded.clean_df.reset_index(drop=True)
    result = cleaned_df.reset_index(drop=True)
    if index_col in dirty.columns and index_col in clean.columns and index_col in result.columns:
        dirty = dirty.set_index(index_col, drop=False)
        clean = clean.set_index(index_col, drop=False)
        result = result.set_index(index_col, drop=False)
        common_index = dirty.index.intersection(clean.index).intersection(result.index)
        dirty = dirty.loc[common_index].reset_index(drop=True)
        clean = clean.loc[common_index].reset_index(drop=True)
        result = result.loc[common_index].reset_index(drop=True)
    else:
        n = min(len(cleaned_df), len(encoded.clean_df), len(encoded.dirty_df))
        dirty = dirty.iloc[:n].reset_index(drop=True)
        clean = clean.iloc[:n].reset_index(drop=True)
        result = result.iloc[:n].reset_index(drop=True)
    cols = [c for c in encoded.feature_cols if c in clean.columns and c in result.columns]
    dirty_errors = 0
    repaired = 0
    changed = 0
    correct_changes = 0
    n = len(dirty)
    for col in cols:
        for i in range(n):
            d = _norm(dirty.loc[i, col])
            c = _norm(clean.loc[i, col])
            r = _norm(result.loc[i, col])
            if d != c:
                dirty_errors += 1
                if r == c:
                    repaired += 1
            if r != d:
                changed += 1
                if r == c:
                    correct_changes += 1
    precision = correct_changes / changed if changed else 0.0
    recall = repaired / dirty_errors if dirty_errors else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "cell_dirty_errors": float(dirty_errors),
        "cell_changed": float(changed),
        "repair_precision": precision,
        "repair_recall": recall,
        "repair_f1": f1,
    }


def _classification_score(X_train, y_train, X_test, y_test, model_type: str) -> float:
    if model_type == "svm":
        model = SVC(kernel="linear", max_iter=10000, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
    model.fit(X_train, y_train.astype(int))
    pred = model.predict(X_test)
    return float(f1_score(y_test.astype(int), pred.astype(int), average="macro", zero_division=0))


def _regression_score(X_train, y_train, X_test, y_test, model_type: str) -> float:
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "ridge":
        model = Ridge()
    else:
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return float(r2_score(y_test, pred))


def _fill_nan(X: np.ndarray) -> np.ndarray:
    out = X.copy()
    means = np.zeros(out.shape[1], dtype=float)
    for col in range(out.shape[1]):
        valid = out[:, col][~np.isnan(out[:, col])]
        means[col] = float(valid.mean()) if len(valid) else 0.0
    for col in range(out.shape[1]):
        mask = np.isnan(out[:, col])
        if mask.any():
            out[mask, col] = means[col]
    return out


def _norm(value) -> str:
    if value is None:
        return ""
    return str(value).strip()
