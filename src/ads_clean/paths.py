from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_root() -> Path:
    return project_root() / "data"


def default_output_root() -> Path:
    return project_root() / "outputs"

