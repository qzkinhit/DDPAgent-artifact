from __future__ import annotations

from dataclasses import dataclass
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from .datasets import DatasetConfig
from .repair_sources import CleanedValueSource, diff_candidate_repairs
from .uniclean_profiles import build_cleaners


@dataclass
class UniCleanResult:
    cleaned_df: pd.DataFrame
    cleaned_csv: Path
    candidate_repairs: Dict[Tuple[int, str], object]
    value_source: CleanedValueSource
    trace: Dict[str, object]


def run_uniclean(
    config: DatasetConfig,
    dirty_csv: Path,
    run_dir: Path,
    single_max: int = 10000,
    verbose: bool = False,
) -> UniCleanResult:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import monotonically_increasing_id
    from Clean import CleanonLocalWithnoSmple

    if not config.cleaner_profile:
        raise ValueError(f"Dataset {config.name} has no UniClean cleaner profile")

    run_dir.mkdir(parents=True, exist_ok=True)
    table_path = run_dir / "uniclean_workflow"
    table_path.mkdir(parents=True, exist_ok=True)

    dirty_df = pd.read_csv(dirty_csv, dtype=str, keep_default_na=False)
    cleaners = build_cleaners(config.cleaner_profile)

    log_path = run_dir / "uniclean_stdout.log"

    def _run():
        spark = (
            SparkSession.builder.appName(f"ADSClean-UniClean-{config.name}")
            .config("spark.executor.memory", "4g")
            .config("spark.driver.memory", "4g")
            .config("spark.sql.shuffle.partitions", "16")
            .config("spark.sql.caseSensitive", "true")
            .getOrCreate()
        )
        try:
            data = spark.read.csv(str(dirty_csv), header=True, inferSchema=False)
            if config.index_col not in data.columns:
                data = data.withColumn(config.index_col, monotonically_increasing_id())
            if config.index_col != "index" and config.index_col in data.columns and "index" not in data.columns:
                data = data.withColumnRenamed(config.index_col, "index")
            fill_cols = [c for c in data.columns if c != "index"]
            if fill_cols:
                data = data.fillna("empty", subset=fill_cols)
            data.persist()

            cleaned_spark = CleanonLocalWithnoSmple(
                spark, cleaners, data, str(table_path), single_max=single_max
            )
            return cleaned_spark.toPandas()
        finally:
            spark.stop()

    if verbose:
        cleaned_df = _run()
    else:
        with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
            cleaned_df = _run()

    if "index" in cleaned_df.columns and config.index_col != "index" and config.index_col not in cleaned_df.columns:
        cleaned_df = cleaned_df.rename(columns={"index": config.index_col})
    cleaned_csv = run_dir / "uniclean_cleaned.csv"
    cleaned_df.to_csv(cleaned_csv, index=False)

    candidates = diff_candidate_repairs(dirty_df, cleaned_df, config.index_col)
    value_source = CleanedValueSource.from_df(
        cleaned_df,
        cleaned_csv,
        index_col=config.index_col,
        source_name="uniclean_runtime",
    )
    trace = {
        "dataset": config.name,
        "cleaner_profile": config.cleaner_profile,
        "cleaner_count": len(cleaners),
        "candidate_repairs": len(candidates),
        "workflow_dir": str(table_path),
        "cleaned_csv": str(cleaned_csv),
        "stdout_log": str(log_path),
    }
    return UniCleanResult(cleaned_df, cleaned_csv, candidates, value_source, trace)


def load_cached_uniclean(
    config: DatasetConfig,
    dirty_df: pd.DataFrame,
    run_dir: Path,
) -> UniCleanResult:
    if not config.cached_uniclean_path:
        raise ValueError(f"Dataset {config.name} has no cached UniClean path")
    cleaned_csv = Path(config.cached_uniclean_path)
    if not cleaned_csv.exists():
        raise FileNotFoundError(cleaned_csv)
    cleaned_df = pd.read_csv(cleaned_csv, dtype=str, keep_default_na=False)
    value_source = CleanedValueSource.from_df(
        cleaned_df,
        cleaned_csv,
        index_col=config.index_col,
        source_name="cached_uniclean",
    )
    candidates = diff_candidate_repairs(dirty_df, cleaned_df, config.index_col)
    trace = {
        "dataset": config.name,
        "scenario": config.scenario,
        "error_rate": config.error_rate,
        "cleaner_profile": config.cleaner_profile,
        "candidate_repairs": len(candidates),
        "cached": True,
        "cleaned_csv": str(cleaned_csv),
    }
    return UniCleanResult(cleaned_df, cleaned_csv, candidates, value_source, trace)
