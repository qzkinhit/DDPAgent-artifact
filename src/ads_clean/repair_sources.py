from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd


@dataclass
class CleanedValueSource:
    cleaned_df: pd.DataFrame
    source_path: Optional[Path]
    index_col: str = "index"
    source_name: str = "cached_uniclean"

    @classmethod
    def from_csv(cls, path: Path, index_col: str = "index", source_name: str = "cached_uniclean") -> "CleanedValueSource":
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df = normalize_index_column(df, index_col)
        return cls(df, path, index_col=index_col, source_name=source_name)

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        source_path: Optional[Path],
        index_col: str = "index",
        source_name: str = "uniclean_runtime",
    ) -> "CleanedValueSource":
        return cls(normalize_index_column(df, index_col), source_path, index_col=index_col, source_name=source_name)

    def value_for(self, row_key: object, column: str):
        if column not in self.cleaned_df.columns:
            raise KeyError(f"Column '{column}' not found in {self.source_name}: {self.source_path}")
        if self.index_col not in self.cleaned_df.columns:
            raise KeyError(f"Index column '{self.index_col}' not found in {self.source_name}: {self.source_path}")
        key = normalize_index_value(row_key)
        indexed = self.cleaned_df.set_index(self.index_col, drop=False)
        if key not in indexed.index:
            raise KeyError(f"Index '{key}' not found in {self.source_name}: {self.source_path}")
        value = indexed.loc[key, column]
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        return value

    def project_to(self, reference_df: pd.DataFrame) -> pd.DataFrame:
        reference = normalize_index_column(reference_df, self.index_col)
        if self.index_col not in reference.columns:
            raise KeyError(f"Reference dataframe has no '{self.index_col}' column")
        if self.index_col not in self.cleaned_df.columns:
            raise KeyError(f"Cleaned dataframe has no '{self.index_col}' column")
        indexed = self.cleaned_df.set_index(self.index_col, drop=False)
        wanted = reference[self.index_col].astype(str).map(normalize_index_value).tolist()
        available = [idx for idx in wanted if idx in indexed.index]
        if not available:
            raise ValueError(f"No overlapping indices in cleaned source: {self.source_path}")
        return indexed.loc[available].reset_index(drop=True)


def normalize_index_column(df: pd.DataFrame, index_col: str = "index") -> pd.DataFrame:
    out = df.copy()
    if index_col not in out.columns and "tno" in out.columns:
        out = out.rename(columns={"tno": index_col})
        numeric = pd.to_numeric(out[index_col], errors="coerce")
        if numeric.notna().any() and numeric.min() == 0:
            out[index_col] = (numeric.astype("Int64") + 1).astype(str)
    if index_col in out.columns:
        out[index_col] = out[index_col].map(normalize_index_value)
    return out


def normalize_index_value(value) -> str:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return str(value).strip()
    if number.is_integer():
        return str(int(number))
    return str(value).strip()


def diff_candidate_repairs(
    dirty_df: pd.DataFrame, cleaned_df: pd.DataFrame, index_col: str = "index"
) -> Dict[Tuple[int, str], object]:
    dirty = normalize_index_column(dirty_df.reset_index(drop=True), index_col)
    cleaned = normalize_index_column(cleaned_df.reset_index(drop=True), index_col)
    if index_col in dirty.columns and index_col in cleaned.columns:
        cleaned_indexed = cleaned.set_index(index_col, drop=False)
        rows = []
        for pos, row in dirty.iterrows():
            key = normalize_index_value(row[index_col])
            if key in cleaned_indexed.index:
                matched = cleaned_indexed.loc[key]
                if isinstance(matched, pd.DataFrame):
                    matched = matched.iloc[0]
                rows.append((pos, row, matched))
    else:
        n = min(len(dirty), len(cleaned))
        rows = [(pos, dirty.loc[pos], cleaned.loc[pos]) for pos in range(n)]

    candidates: Dict[Tuple[int, str], object] = {}
    common_cols = [c for c in dirty.columns if c in cleaned.columns]
    for pos, dirty_row, cleaned_row in rows:
        row_id = pos
        if index_col in dirty.columns:
            try:
                row_id = int(float(dirty_row[index_col]))
            except (TypeError, ValueError):
                row_id = pos
        for col in common_cols:
            if col == index_col:
                continue
            old = dirty_row[col]
            new = cleaned_row[col]
            if _norm_cell(old) != _norm_cell(new):
                candidates[(row_id, col)] = new
                candidates[(pos, col)] = new
    return candidates


def _norm_cell(value) -> str:
    if value is None:
        return ""
    return str(value).strip()
