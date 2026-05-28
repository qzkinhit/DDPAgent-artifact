from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .datasets import DatasetConfig, canonical_dataset_name
from .paths import project_root


ERROR_RATES: Sequence[str] = ("025", "05", "075", "1", "125", "15", "175", "2")
RESULT_ASSET_DIR = project_root() / "result_assets"


@dataclass(frozen=True)
class ResultDatasetSpec:
    name: str
    original_id: str
    artificial_id: str
    uniclean_prefix: str
    dirty_stem: str
    task_type: str
    model_type: str
    target: str
    cleaner_profile: str
    protected_cols: Sequence[str]
    derived_target: bool = False
    large: bool = False
    cluster_cols: Sequence[str] = ()


SPECS: Dict[str, ResultDatasetSpec] = {
    "hospitals": ResultDatasetSpec(
        name="hospitals",
        original_id="1_hospital",
        artificial_id="1_hospitals",
        uniclean_prefix="1_hospital",
        dirty_stem="hospitals",
        task_type="classification",
        model_type="random_forest",
        target="MeasureCode",
        cleaner_profile="hospitals",
        protected_cols=("MeasureCode",),
    ),
    "flights": ResultDatasetSpec(
        name="flights",
        original_id="2_flights",
        artificial_id="2_flights",
        uniclean_prefix="2_flights",
        dirty_stem="flights",
        task_type="classification",
        model_type="random_forest",
        target="arrival_delay_bucket",
        cleaner_profile="flights",
        protected_cols=("arrival_delay_bucket",),
        derived_target=True,
    ),
    "beers": ResultDatasetSpec(
        name="beers",
        original_id="3_beers",
        artificial_id="3_beers",
        uniclean_prefix="3_beers",
        dirty_stem="beers",
        task_type="classification",
        model_type="random_forest",
        target="style",
        cleaner_profile="beers",
        protected_cols=("style",),
    ),
    "rayyan": ResultDatasetSpec(
        name="rayyan",
        original_id="4_rayyan",
        artificial_id="4_rayyan",
        uniclean_prefix="4_rayyan",
        dirty_stem="rayyan",
        task_type="classification",
        model_type="random_forest",
        target="article_language",
        cleaner_profile="rayyan",
        protected_cols=("article_language",),
    ),
    "tax": ResultDatasetSpec(
        name="tax",
        original_id="5_tax",
        artificial_id="5_tax",
        uniclean_prefix="5_tax",
        dirty_stem="tax",
        task_type="regression",
        model_type="ridge",
        target="rate",
        cleaner_profile="tax",
        protected_cols=("rate",),
        large=True,
        cluster_cols=("zip",),
    ),
}


def find_uniclean_result_root(explicit: Optional[Path] = None) -> Path:
    return _first_existing_root(
        explicit,
        os.environ.get("ADS_UNICLEAN_RESULT_ROOT"),
        RESULT_ASSET_DIR / "UnicleanResult",
        Path("/Users/qianzekai/PycharmProjects/UnicleanResult"),
        label="UniCleanResult",
    )


def find_legacy_benchmark_root(explicit: Optional[Path] = None) -> Path:
    return _first_existing_root(
        explicit,
        os.environ.get("DEMANDPREP_LEGACY_BENCHMARK_ROOT"),
        RESULT_ASSET_DIR / "DemandPrep-Benchmark",
        Path("/Users/qianzekai/PycharmProjects/DemandPrep-Benchmark"),
        label="DemandPrep-Benchmark",
    )


def _first_existing_root(explicit, env_value, repo_path: Path, external_path: Path, label: str) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if env_value:
        candidates.append(Path(env_value))
    candidates.extend([repo_path, external_path])
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"{label} root not found. Checked: {', '.join(map(str, candidates))}")


def result_dataset_names() -> Sequence[str]:
    return ("hospitals", "flights", "beers", "rayyan", "tax")


def build_result_dataset_config(
    dataset: str,
    scenario: str = "original",
    error_rate: Optional[str] = None,
    result_root: Optional[Path] = None,
    subset_policy: str = "cluster10k",
) -> DatasetConfig:
    name = canonical_dataset_name(dataset)
    if name not in SPECS:
        raise KeyError(f"Unknown UniCleanResult dataset: {dataset}")
    if scenario not in {"original", "artificial"}:
        raise ValueError("scenario must be 'original' or 'artificial'")
    spec = SPECS[name]
    root = find_uniclean_result_root(result_root)

    dirty_path: Path
    clean_path: Path
    subset_index_path: Optional[Path] = None
    subset_size: Optional[int] = None
    subset_key_cols: Sequence[str] = ()
    subset_source = ""

    if scenario == "original":
        base = root / "datasets_and_rules" / "original_datasets" / spec.original_id
        dirty_path = base / "dirty_index.csv"
        clean_path = base / "clean_index.csv"
        if name == "tax" and subset_policy.startswith("cluster"):
            dirty_path = base / "subset_dirty_index_10k.csv"
            clean_path = base / "subset_clean_index_10k.csv"
            subset_source = "existing_cluster_subset_10k"
        cached = _cached_uniclean_path(root, spec, scenario, error_rate=None)
    else:
        if error_rate is None:
            raise ValueError("Artificial scenarios require --error-rate")
        rate_dir = rate_to_dir(error_rate)
        base = root / "datasets_and_rules" / "artificial_error_datasets" / spec.artificial_id
        dirty_path = base / f"dirty_mixed_{rate_dir}" / f"dirty_{spec.dirty_stem}_mix_{rate_dir}.csv"
        if name in {"tax"}:
            clean_path = base / f"dirty_mixed_{rate_dir}" / f"dirty_{spec.dirty_stem}.csv"
            subset_index_path = base / "subset_tax_10k" / "subset_tax_10k_clean_index.csv"
            subset_source = "existing_tax_subset_10k"
        else:
            clean_path = base / "clean_index.csv"
        cached = _cached_uniclean_path(root, spec, scenario, error_rate=error_rate)

    return DatasetConfig(
        name=name,
        source="uniclean_result",
        dirty_path=dirty_path,
        clean_path=clean_path,
        task_type=spec.task_type,
        model_type=spec.model_type,
        target=spec.target,
        index_col="index",
        cleaner_profile=spec.cleaner_profile,
        derived_target=spec.derived_target,
        protected_cols=tuple(spec.protected_cols),
        notes=f"{scenario} scenario from UniCleanResult; subset={subset_source or 'full'}",
        scenario=scenario,
        error_rate=error_rate,
        cached_uniclean_path=cached,
        subset_index_path=subset_index_path,
        subset_size=subset_size,
        subset_key_cols=tuple(subset_key_cols),
        subset_source=subset_source,
    )


def _cached_uniclean_path(
    root: Path,
    spec: ResultDatasetSpec,
    scenario: str,
    error_rate: Optional[str],
) -> Path:
    if scenario == "original":
        base = root / "Uniclean_cleaned_data" / "original_error_cleaned_data"
        if spec.name == "tax":
            return base / "5_tax50k_cleaned_by_uniclean.csv"
        return base / f"{spec.uniclean_prefix}_cleaned_by_uniclean.csv"
    if error_rate is None:
        raise ValueError("error_rate is required for artificial cached UniClean path")
    base = root / "Uniclean_cleaned_data" / "artificial_error_cleaned_data"
    return base / f"{spec.uniclean_prefix}_{error_rate}_cleaned_by_uniclean.csv"


def rate_to_dir(rate: str) -> str:
    mapping = {
        "025": "0.25",
        "05": "0.5",
        "075": "0.75",
        "1": "1",
        "125": "1.25",
        "15": "1.5",
        "175": "1.75",
        "2": "2",
    }
    if rate not in mapping:
        raise ValueError(f"Unknown artificial error rate: {rate}")
    return mapping[rate]


def baseline_cleaned_candidates(
    dataset: str,
    system: str,
    scenario: str,
    error_rate: Optional[str],
    result_root: Optional[Path] = None,
) -> List[Path]:
    name = canonical_dataset_name(dataset)
    spec = SPECS[name]
    root = find_uniclean_result_root(result_root)
    if scenario == "original":
        base = root / "baseline_cleaned_data" / "original_cleaned_data" / system
        patterns = [
            f"{spec.original_id}_cleaned_by_{system}.csv",
            f"{spec.original_id}*cleaned_by_{system}.csv",
            f"{spec.artificial_id}*cleaned_by_{system}.csv",
        ]
    else:
        if error_rate is None:
            raise ValueError("Artificial baseline lookup requires error_rate")
        base = root / "baseline_cleaned_data" / "artificial_error_cleaned_data" / system
        patterns = [
            f"{spec.artificial_id}_{error_rate}_cleaned_by_{system}.csv",
            f"{spec.uniclean_prefix}_{error_rate}_cleaned_by_{system}.csv",
            f"{spec.original_id}_{error_rate}_cleaned_by_{system}.csv",
        ]
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(sorted(base.glob(pattern)))
    unique: List[Path] = []
    seen = set()
    for path in matches:
        if path not in seen and path.exists():
            unique.append(path)
            seen.add(path)
    return unique


def scan_asset_catalog(uniclean_root: Optional[Path] = None, legacy_root: Optional[Path] = None) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    uroot = find_uniclean_result_root(uniclean_root)
    legacy_policy_root = find_legacy_benchmark_root(legacy_root)

    for scenario, rel in [
        ("original", "Uniclean_cleaned_data/original_error_cleaned_data"),
        ("artificial", "Uniclean_cleaned_data/artificial_error_cleaned_data"),
        ("baseline_original", "baseline_cleaned_data/original_cleaned_data"),
        ("baseline_artificial", "baseline_cleaned_data/artificial_error_cleaned_data"),
    ]:
        base = uroot / rel
        for path in sorted(base.glob("**/*.csv")):
            rows.append(_csv_asset_row(path, uroot, scenario, "uniclean_result"))

    for path in sorted((legacy_policy_root / "results_and_logs" / "summary").glob("*.csv")):
        rows.append(_csv_asset_row(path, legacy_policy_root, "legacy_summary", "legacy_policy_benchmark"))

    return pd.DataFrame(rows)


def _csv_asset_row(path: Path, root: Path, scenario: str, provenance: str) -> Dict[str, object]:
    rel = path.relative_to(root)
    row_count = None
    columns = ""
    index_col = ""
    try:
        df0 = pd.read_csv(path, nrows=0, dtype=str, keep_default_na=False)
        columns = "|".join(str(c) for c in df0.columns)
        if "index" in df0.columns:
            index_col = "index"
        elif "tno" in df0.columns:
            index_col = "tno"
        with path.open("rb") as f:
            row_count = max(0, sum(1 for _ in f) - 1)
    except Exception as exc:  # pragma: no cover - catalog should keep going
        columns = f"ERROR:{exc}"
    return {
        "path": str(path),
        "relative_path": str(rel),
        "scenario": scenario,
        "provenance": provenance,
        "row_count": row_count,
        "index_col": index_col,
        "columns": columns,
        "bytes": path.stat().st_size if path.exists() else None,
    }


def copy_result_snapshots(
    uniclean_source: Path,
    legacy_source: Path,
    destination: Optional[Path] = None,
) -> Path:
    dest = Path(destination) if destination else RESULT_ASSET_DIR
    dest.mkdir(parents=True, exist_ok=True)
    _copytree_update(Path(uniclean_source), dest / "UnicleanResult")
    _copytree_update(Path(legacy_source), dest / "DemandPrep-Benchmark")
    return dest


def _copytree_update(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copytree(src, dst, dirs_exist_ok=True)
