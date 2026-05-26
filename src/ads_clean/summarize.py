from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def summarize_adsclean_runs(output_root: Path) -> Path:
    root = Path(output_root)
    rows: List[Dict[str, object]] = []
    metrics_paths = sorted(root.glob("adsclean/**/metrics.json"))
    if not metrics_paths:
        metrics_paths = sorted(root.glob("**/metrics.json"))
    for metrics_path in metrics_paths:
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                row = json.load(f)
            row["run_dir"] = str(metrics_path.parent)
            rows.append(row)
        except Exception as exc:  # pragma: no cover
            rows.append({"run_dir": str(metrics_path.parent), "status": "error", "error": str(exc)})
    summary_path = root / "adsclean_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return summary_path
