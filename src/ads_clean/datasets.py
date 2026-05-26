from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .paths import default_data_root


MISSING_TOKENS = {
    "",
    "empty",
    "Empty",
    "EMPTY",
    "nan",
    "NaN",
    "NULL",
    "null",
    "None",
    "__NULL__",
}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    source: str
    dirty_path: Path
    clean_path: Optional[Path]
    task_type: str
    model_type: str
    target: str
    index_col: str = "index"
    cleaner_profile: str = ""
    derived_target: bool = False
    drop_cols: Tuple[str, ...] = ("index", "id")
    protected_cols: Tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""
    scenario: str = "original"
    error_rate: Optional[str] = None
    cached_uniclean_path: Optional[Path] = None
    subset_index_path: Optional[Path] = None
    subset_size: Optional[int] = None
    subset_key_cols: Tuple[str, ...] = field(default_factory=tuple)
    subset_source: str = ""

    @property
    def has_clean_reference(self) -> bool:
        return self.clean_path is not None and self.clean_path.exists()


def _p(root: Path, *parts: str) -> Path:
    return root.joinpath(*parts)


def default_dataset_configs(data_root: Optional[Path] = None) -> Dict[str, DatasetConfig]:
    root = Path(data_root) if data_root else default_data_root()
    u = root / "uniclean"
    d = root / "demandclean"
    configs = {
        "beers": DatasetConfig(
            name="beers",
            source="shared",
            dirty_path=_p(u, "3_beers", "dirty_index.csv"),
            clean_path=_p(u, "3_beers", "clean_index.csv"),
            task_type="classification",
            model_type="random_forest",
            target="style",
            cleaner_profile="beers",
            protected_cols=("style",),
            notes="Natural overlap between DemandClean and UniClean.",
        ),
        "hospitals": DatasetConfig(
            name="hospitals",
            source="uniclean",
            dirty_path=_p(u, "1_hospitals", "dirty_index.csv"),
            clean_path=_p(u, "1_hospitals", "clean_index.csv"),
            task_type="classification",
            model_type="random_forest",
            target="MeasureCode",
            cleaner_profile="hospitals",
            protected_cols=("MeasureCode",),
            notes="Governance benchmark task: hospital measure-code classification.",
        ),
        "flights": DatasetConfig(
            name="flights",
            source="uniclean",
            dirty_path=_p(u, "2_flights", "dirty_index.csv"),
            clean_path=_p(u, "2_flights", "clean_index.csv"),
            task_type="classification",
            model_type="random_forest",
            target="arrival_delay_bucket",
            cleaner_profile="flights",
            derived_target=True,
            protected_cols=("arrival_delay_bucket",),
            notes="Derived governance task from scheduled and actual arrival times.",
        ),
        "rayyan": DatasetConfig(
            name="rayyan",
            source="uniclean",
            dirty_path=_p(u, "4_rayyan", "dirty_index.csv"),
            clean_path=_p(u, "4_rayyan", "clean_index.csv"),
            task_type="classification",
            model_type="random_forest",
            target="article_language",
            cleaner_profile="rayyan",
            protected_cols=("article_language",),
            notes="Governance benchmark task: article language classification.",
        ),
        "tax": DatasetConfig(
            name="tax",
            source="uniclean",
            dirty_path=_p(u, "5_tax", "dirty_index.csv"),
            clean_path=_p(u, "5_tax", "clean_index.csv"),
            task_type="regression",
            model_type="ridge",
            target="rate",
            cleaner_profile="tax",
            protected_cols=("rate",),
            notes="Default packaged version uses the 10k subset.",
        ),
    }

    for path in (d.iterdir() if d.exists() else []):
        if not path.is_dir() or path.name in configs:
            continue
        dirty = path / "dirty_index.csv"
        clean = path / "clean_index.csv"
        if dirty.exists():
            # These are available for DemandClean-only experiments. The
            # orchestrated ADS pipeline focuses on UniClean-compatible tables.
            configs[f"demandclean:{path.name}"] = DatasetConfig(
                name=f"demandclean:{path.name}",
                source="demandclean",
                dirty_path=dirty,
                clean_path=clean if clean.exists() else None,
                task_type="classification",
                model_type="random_forest",
                target="",
                cleaner_profile="",
                notes="Packaged DemandClean benchmark data; no UniClean profile by default.",
            )
    return configs


def load_dataset_config(name: str, data_root: Optional[Path] = None) -> DatasetConfig:
    configs = default_dataset_configs(data_root)
    if name not in configs:
        available = ", ".join(sorted(configs))
        raise KeyError(f"Unknown dataset '{name}'. Available datasets: {available}")
    cfg = configs[name]
    if not cfg.dirty_path.exists():
        raise FileNotFoundError(f"Dirty dataset not found: {cfg.dirty_path}")
    return cfg


def parse_fd_rules(path: Optional[Path]) -> List[Tuple[str, str]]:
    if path is None or not path.exists():
        return []
    rules: List[Tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "⇒" in line:
            lhs, rhs = line.split("⇒", 1)
        elif "->" in line:
            lhs, rhs = line.split("->", 1)
        else:
            continue
        lhs = lhs.strip().replace("，", ",")
        rhs = rhs.strip()
        if lhs and rhs:
            rules.append((lhs, rhs))
    return rules


def packaged_suite() -> Sequence[str]:
    return ("beers", "hospitals", "flights", "rayyan", "tax")


def canonical_dataset_name(name: str) -> str:
    aliases = {
        "hospital": "hospitals",
        "hospitals": "hospitals",
        "1_hospital": "hospitals",
        "1_hospitals": "hospitals",
        "flight": "flights",
        "flights": "flights",
        "2_flights": "flights",
        "beer": "beers",
        "beers": "beers",
        "3_beers": "beers",
        "rayyan": "rayyan",
        "4_rayyan": "rayyan",
        "tax": "tax",
        "5_tax": "tax",
    }
    return aliases.get(name, name)
