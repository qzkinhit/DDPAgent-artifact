from pathlib import Path

from demandprep.datasets import load_dataset_config
from demandprep.preprocess import prepare_dataset


def test_beers_preprocess(tmp_path):
    cfg = load_dataset_config("beers")
    enc = prepare_dataset(cfg, tmp_path)
    assert enc.X_dirty.shape[0] == len(enc.y_dirty)
    assert "style" not in enc.feature_cols
    assert enc.work_dirty_csv.exists()


def test_flights_derived_label(tmp_path):
    cfg = load_dataset_config("flights")
    enc = prepare_dataset(cfg, tmp_path)
    assert cfg.target == "arrival_delay_bucket"
    assert enc.X_dirty.shape[0] > 0
    assert enc.dropped_rows > 0

