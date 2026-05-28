from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .datasets import DatasetConfig, MISSING_TOKENS


@dataclass
class EncodedDataset:
    config: DatasetConfig
    dirty_df: pd.DataFrame
    clean_df: Optional[pd.DataFrame]
    feature_cols: List[str]
    categorical_cols: Set[str]
    label_encoder: Optional[LabelEncoder]
    feature_encoders: Dict[str, LabelEncoder]
    scaler: StandardScaler
    X_dirty: np.ndarray
    y_dirty: np.ndarray
    X_clean: Optional[np.ndarray]
    y_clean: Optional[np.ndarray]
    work_dirty_csv: Path
    work_clean_csv: Optional[Path]
    dropped_rows: int

    @property
    def csv_columns(self) -> List[str]:
        return list(self.dirty_df.columns)

    def decode_feature_value(self, feature_col: str, encoded_value) -> object:
        if encoded_value is None:
            return np.nan
        try:
            val = float(encoded_value)
        except (TypeError, ValueError):
            return encoded_value
        if np.isnan(val):
            return np.nan
        col_idx = self.feature_cols.index(feature_col)
        raw_numeric = val * self.scaler.scale_[col_idx] + self.scaler.mean_[col_idx]
        if feature_col in self.feature_encoders:
            le = self.feature_encoders[feature_col]
            if len(le.classes_) == 0:
                return np.nan
            class_idx = int(round(raw_numeric))
            class_idx = max(0, min(class_idx, len(le.classes_) - 1))
            return le.inverse_transform([class_idx])[0]
        return raw_numeric

    def encode_features(self, df: pd.DataFrame) -> np.ndarray:
        encoded = _encode_feature_frame(df, self.feature_cols, self.categorical_cols, self.feature_encoders)
        return _scale_with_nan(encoded, self.scaler)


def prepare_dataset(config: DatasetConfig, work_dir: Path) -> EncodedDataset:
    work_dir.mkdir(parents=True, exist_ok=True)
    dirty = _normalize_index_column(_read_csv(config.dirty_path), config.index_col)
    clean = _normalize_index_column(_read_csv(config.clean_path), config.index_col) if config.clean_path and config.clean_path.exists() else None
    dirty, clean = _apply_subset(dirty, clean, config)
    if clean is not None and len(clean) != len(dirty):
        n = min(len(dirty), len(clean))
        dirty = dirty.iloc[:n].reset_index(drop=True)
        clean = clean.iloc[:n].reset_index(drop=True)

    dirty = _add_or_normalize_target(dirty, config)
    if clean is not None:
        clean = _add_or_normalize_target(clean, config)

    valid_mask = _valid_target_mask(dirty[config.target], config.task_type)
    if clean is not None and config.target in clean.columns:
        valid_mask &= _valid_target_mask(clean[config.target], config.task_type)
    dropped = int((~valid_mask).sum())
    mask_values = valid_mask.to_numpy()
    dirty = dirty.iloc[mask_values].reset_index(drop=True)
    clean = clean.iloc[mask_values].reset_index(drop=True) if clean is not None else None

    feature_cols = _feature_columns(dirty, config)
    categorical_cols = _detect_categorical_columns(dirty, feature_cols)

    y_dirty, label_encoder = _encode_target(dirty[config.target], config.task_type)
    y_clean = None
    if clean is not None and config.target in clean.columns:
        y_clean, _ = _encode_target(clean[config.target], config.task_type, label_encoder)

    feature_encoders = _fit_feature_encoders(dirty, feature_cols, categorical_cols)
    X_dirty_raw = _encode_feature_frame(dirty, feature_cols, categorical_cols, feature_encoders)
    scaler = _fit_scaler_with_nan(X_dirty_raw)
    X_dirty = _scale_with_nan(X_dirty_raw, scaler)

    X_clean = None
    if clean is not None:
        X_clean_raw = _encode_feature_frame(clean, feature_cols, categorical_cols, feature_encoders)
        X_clean = _scale_with_nan(X_clean_raw, scaler)

    work_dirty = work_dir / f"{config.name}_dirty_work.csv"
    work_clean = work_dir / f"{config.name}_clean_work.csv" if clean is not None else None
    dirty.to_csv(work_dirty, index=False)
    if clean is not None and work_clean is not None:
        clean.to_csv(work_clean, index=False)

    return EncodedDataset(
        config=config,
        dirty_df=dirty,
        clean_df=clean,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        label_encoder=label_encoder,
        feature_encoders=feature_encoders,
        scaler=scaler,
        X_dirty=X_dirty,
        y_dirty=y_dirty,
        X_clean=X_clean,
        y_clean=y_clean,
        work_dirty_csv=work_dirty,
        work_clean_csv=work_clean,
        dropped_rows=dropped,
    )


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [str(c).strip().strip("\ufeff") for c in df.columns]
    return df


def _normalize_index_column(df: pd.DataFrame, index_col: str) -> pd.DataFrame:
    df = df.copy()
    if index_col not in df.columns and "tno" in df.columns:
        df = df.rename(columns={"tno": index_col})
        numeric = pd.to_numeric(df[index_col], errors="coerce")
        if numeric.notna().any() and numeric.min() == 0:
            df[index_col] = (numeric.astype("Int64") + 1).astype(str)
    if index_col in df.columns:
        df[index_col] = df[index_col].map(_normalize_index_value)
    return df


def _normalize_index_value(value) -> str:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return str(value).strip()
    if number.is_integer():
        return str(int(number))
    return str(value).strip()


def _apply_subset(
    dirty: pd.DataFrame,
    clean: Optional[pd.DataFrame],
    config: DatasetConfig,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if config.subset_index_path and config.subset_index_path.exists():
        subset_df = _normalize_index_column(_read_csv(config.subset_index_path), config.index_col)
        if config.index_col not in subset_df.columns:
            raise ValueError(f"Subset file has no {config.index_col} column: {config.subset_index_path}")
        wanted = subset_df[config.index_col].astype(str).tolist()
        return _filter_by_index(dirty, wanted, config.index_col), (
            _filter_by_index(clean, wanted, config.index_col) if clean is not None else None
        )

    if config.subset_size and config.subset_key_cols:
        source = clean if clean is not None else dirty
        wanted = _clustered_index_subset(source, config.index_col, config.subset_key_cols, config.subset_size)
        return _filter_by_index(dirty, wanted, config.index_col), (
            _filter_by_index(clean, wanted, config.index_col) if clean is not None else None
        )

    return dirty, clean


def _filter_by_index(df: Optional[pd.DataFrame], wanted: List[str], index_col: str) -> pd.DataFrame:
    if df is None:
        raise ValueError("Cannot apply an index subset to a missing dataframe")
    if index_col not in df.columns:
        raise ValueError(f"Dataframe has no {index_col} column for subset filtering")
    normalized = df.copy()
    normalized[index_col] = normalized[index_col].astype(str).map(_normalize_index_value)
    indexed = normalized.set_index(index_col, drop=False)
    available = [idx for idx in wanted if idx in indexed.index]
    if not available:
        raise ValueError("Subset filtering removed all rows")
    return indexed.loc[available].reset_index(drop=True)


def _clustered_index_subset(df: pd.DataFrame, index_col: str, key_cols: Tuple[str, ...], subset_size: int) -> List[str]:
    available_keys = [col for col in key_cols if col in df.columns]
    if index_col not in df.columns:
        raise ValueError(f"Cannot build clustered subset without {index_col}")
    if not available_keys:
        return df[index_col].astype(str).head(subset_size).tolist()

    keyed = df.copy()
    keyed[index_col] = keyed[index_col].astype(str).map(_normalize_index_value)
    grouped = keyed.groupby(available_keys, dropna=False, sort=True)[index_col].apply(list)
    groups = sorted(
        (sorted(values, key=_index_sort_key) for values in grouped),
        key=lambda values: (-len(values), _index_sort_key(values[0]) if values else (0, "")),
    )

    selected: List[str] = []
    seen = set()
    for group in groups:
        for idx in group:
            if idx in seen:
                continue
            selected.append(idx)
            seen.add(idx)
            if len(selected) >= subset_size:
                return selected
    return selected


def _index_sort_key(value: str) -> Tuple[int, str]:
    try:
        return int(float(value)), str(value)
    except (TypeError, ValueError):
        return 0, str(value)


def _add_or_normalize_target(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    df = df.copy()
    if config.name == "flights":
        df[config.target] = _flight_delay_bucket(df)
    elif config.name == "rayyan" and config.target in df.columns:
        df[config.target] = df[config.target].map(_normalize_language)
    return df


def _flight_delay_bucket(df: pd.DataFrame) -> List[Optional[str]]:
    labels: List[Optional[str]] = []
    for _, row in df.iterrows():
        sched = _parse_time(row.get("sched_arr_time", ""))
        actual = _parse_time(row.get("act_arr_time", ""))
        if sched is None or actual is None:
            labels.append(None)
            continue
        diff = (actual - sched).total_seconds() / 60.0
        if diff < -720:
            diff += 1440
        if diff > 720:
            diff -= 1440
        if diff <= -5:
            labels.append("early")
        elif diff <= 15:
            labels.append("on_time")
        elif diff <= 60:
            labels.append("late")
        else:
            labels.append("severely_late")
    return labels


def _parse_time(value) -> Optional[datetime]:
    s = str(value).strip()
    if s in MISSING_TOKENS:
        return None
    for fmt in ("%I:%M %p", "%I:%M%p"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def _normalize_language(value) -> Optional[str]:
    s = str(value).strip()
    if s in MISSING_TOKENS:
        return None
    low = s.lower()
    if "english" in low or low == "eng":
        return "eng"
    if low in {"jpn", "japanese"}:
        return "jpn"
    if low in {"ger", "deu", "german"}:
        return "ger"
    if low in {"fre", "fra", "french"}:
        return "fre"
    if low in {"spa", "spanish"}:
        return "spa"
    return low


def _valid_target_mask(series: pd.Series, task_type: str) -> pd.Series:
    values = series.map(lambda x: None if x is None or str(x).strip() in MISSING_TOKENS else x)
    if task_type == "regression":
        return pd.to_numeric(values, errors="coerce").notna()
    return values.notna()


def _feature_columns(df: pd.DataFrame, config: DatasetConfig) -> List[str]:
    drop = set(config.drop_cols) | {config.target}
    drop |= set(config.protected_cols)
    return [c for c in df.columns if c not in drop]


def _clean_missing(series: pd.Series) -> pd.Series:
    return series.map(lambda x: np.nan if x is None or str(x).strip() in MISSING_TOKENS else x)


def _detect_categorical_columns(df: pd.DataFrame, feature_cols: Iterable[str]) -> Set[str]:
    categorical = set()
    for col in feature_cols:
        values = _clean_missing(df[col]).dropna()
        if values.empty:
            categorical.add(col)
            continue
        converted = pd.to_numeric(values, errors="coerce")
        if converted.isna().any():
            categorical.add(col)
    return categorical


def _encode_target(series: pd.Series, task_type: str, encoder: Optional[LabelEncoder] = None):
    values = _clean_missing(series)
    if task_type == "regression":
        return pd.to_numeric(values, errors="coerce").astype(float).to_numpy(), None
    if encoder is None:
        encoder = LabelEncoder()
        encoder.fit(values.astype(str))
    lookup = {klass: idx for idx, klass in enumerate(encoder.classes_)}
    return values.astype(str).map(lambda value: lookup.get(value, np.nan)).astype(float).to_numpy(), encoder


def _fit_feature_encoders(df: pd.DataFrame, feature_cols: List[str], categorical_cols: Set[str]) -> Dict[str, LabelEncoder]:
    encoders: Dict[str, LabelEncoder] = {}
    for col in feature_cols:
        if col not in categorical_cols:
            continue
        vals = _clean_missing(df[col]).dropna().astype(str)
        le = LabelEncoder()
        if vals.empty:
            le.fit(["__UNK__"])
        else:
            le.fit(sorted(vals.unique()))
        encoders[col] = le
    return encoders


def _encode_feature_frame(
    df: pd.DataFrame,
    feature_cols: List[str],
    categorical_cols: Set[str],
    encoders: Dict[str, LabelEncoder],
) -> np.ndarray:
    out = np.empty((len(df), len(feature_cols)), dtype=float)
    out[:] = np.nan
    for j, col in enumerate(feature_cols):
        values = _clean_missing(df[col])
        if col in categorical_cols:
            le = encoders[col]
            lookup = {klass: idx for idx, klass in enumerate(le.classes_)}
            out[:, j] = values.map(
                lambda value: np.nan
                if value is None or (isinstance(value, float) and np.isnan(value))
                else lookup.get(str(value), np.nan)
            ).astype(float).to_numpy()
        else:
            out[:, j] = pd.to_numeric(values, errors="coerce").astype(float).to_numpy()
    return out


def _fit_scaler_with_nan(X: np.ndarray) -> StandardScaler:
    filled = _fill_nan_by_column(X)
    scaler = StandardScaler()
    scaler.fit(filled)
    scaler.scale_[scaler.scale_ == 0] = 1.0
    return scaler


def _scale_with_nan(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    nan_mask = np.isnan(X)
    filled = _fill_nan_by_column(X)
    transformed = scaler.transform(filled)
    transformed[nan_mask] = np.nan
    return transformed


def _fill_nan_by_column(X: np.ndarray) -> np.ndarray:
    out = X.copy().astype(float)
    means = np.zeros(out.shape[1], dtype=float)
    for j in range(out.shape[1]):
        valid = out[:, j][~np.isnan(out[:, j])]
        means[j] = float(valid.mean()) if len(valid) else 0.0
    for j in range(out.shape[1]):
        mask = np.isnan(out[:, j])
        if mask.any():
            out[mask, j] = means[j]
    return out
