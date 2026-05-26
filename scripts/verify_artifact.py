#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"beers", "flights", "hospitals", "rayyan", "tax"}
BASELINES = {"baran", "bigdansing", "holistic", "holoclean", "horizon"}
REFERENCE_STRATEGIES = {"no_op", "oracle_full_repair", "delete_true_error_rows", "uniclean_full"}


def fail(message: str) -> None:
    raise SystemExit(f"[artifact-check] {message}")


def read_csv(rel: str) -> pd.DataFrame:
    path = ROOT / rel
    if not path.exists():
        fail(f"missing required file: {rel}")
    return pd.read_csv(path)


def check_no_soccer_paths() -> None:
    bad = [p for p in ROOT.rglob("*") if "soccer" in str(p.relative_to(ROOT)).lower()]
    if bad:
        sample = "\n".join(str(p.relative_to(ROOT)) for p in bad[:10])
        fail(f"unexpected Soccer files in paper artifact:\n{sample}")


def check_packaged_data() -> None:
    expected = {
        "data/uniclean/1_hospitals/dirty_index.csv",
        "data/uniclean/2_flights/dirty_index.csv",
        "data/uniclean/3_beers/dirty_index.csv",
        "data/uniclean/4_rayyan/dirty_index.csv",
        "data/uniclean/5_tax/dirty_index.csv",
        "result_assets/UnicleanResult/datasets_and_rules/original_datasets/5_tax/subset_dirty_index_10k.csv",
    }
    missing = [rel for rel in sorted(expected) if not (ROOT / rel).exists()]
    if missing:
        fail("missing packaged data files:\n" + "\n".join(missing))


def check_baseline_assets() -> None:
    base = ROOT / "result_assets" / "UnicleanResult" / "baseline_cleaned_data"
    if not base.exists():
        fail("missing baseline cleaned data directory")
    systems = {p.name for p in (base / "original_cleaned_data").iterdir() if p.is_dir()}
    systems |= {p.name for p in (base / "artificial_error_cleaned_data").iterdir() if p.is_dir()}
    unexpected = systems - BASELINES
    missing = BASELINES - systems
    if unexpected:
        fail(f"unexpected baseline systems: {sorted(unexpected)}")
    if missing:
        fail(f"missing baseline systems: {sorted(missing)}")


def check_summary(rel: str, expected_rows: int | None = None) -> None:
    df = read_csv(rel)
    if "dataset" in df.columns:
        unexpected = set(df["dataset"].dropna()) - DATASETS
        if unexpected:
            fail(f"{rel} contains unexpected datasets: {sorted(unexpected)}")
    if "strategy" in df.columns:
        allowed = BASELINES | REFERENCE_STRATEGIES
        unexpected = set(df["strategy"].dropna()) - allowed
        if unexpected:
            fail(f"{rel} contains unexpected strategies: {sorted(unexpected)}")
    if expected_rows is not None and len(df) != expected_rows:
        fail(f"{rel} has {len(df)} rows, expected {expected_rows}")


def check_paper_summaries() -> None:
    check_summary("outputs/experiments_20260519_final/adsclean/adsclean_summary.csv", 45)
    check_summary("outputs/experiments_20260519_final/baseline_eval/original/baseline_ml_summary.csv", 45)
    check_summary("outputs/experiments_20260519_final/baseline_eval/artificial/baseline_ml_summary.csv", 360)
    check_summary("outputs/experiments_20260520_hospital_measurecode/adsclean/adsclean_summary.csv", 9)
    check_summary("outputs/experiments_20260520_hospital_measurecode/baseline_eval/original/baseline_ml_summary.csv", 9)
    check_summary("outputs/experiments_20260520_hospital_measurecode/baseline_eval/artificial/baseline_ml_summary.csv", 72)
    check_summary("outputs/experiments_20260519_verifier/adsclean/adsclean_summary.csv", 2)


def check_python_defaults() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from ads_clean.datasets import packaged_suite
    from ads_clean.result_assets import result_dataset_names

    if set(packaged_suite()) != DATASETS:
        fail(f"packaged_suite mismatch: {packaged_suite()}")
    if set(result_dataset_names()) != DATASETS:
        fail(f"result_dataset_names mismatch: {result_dataset_names()}")


def check_large_files() -> None:
    too_large = [p for p in ROOT.rglob("*") if p.is_file() and p.stat().st_size >= 95 * 1024 * 1024]
    if too_large:
        sample = "\n".join(f"{p.relative_to(ROOT)} {p.stat().st_size}" for p in too_large[:10])
        fail(f"files near GitHub single-file limit:\n{sample}")


def main() -> int:
    check_no_soccer_paths()
    check_packaged_data()
    check_baseline_assets()
    check_paper_summaries()
    check_python_defaults()
    check_large_files()
    print("[artifact-check] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
