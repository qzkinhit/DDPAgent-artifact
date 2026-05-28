import numpy as np
import pandas as pd

from demandprep.executor import execute_final_cleaning
from demandprep.repair_sources import CleanedValueSource


class DummyConfig:
    index_col = "index"


class DummyEncoded:
    feature_cols = ["a", "b"]
    config = DummyConfig()

    def __init__(self):
        self.dirty_df = pd.DataFrame({"index": [1, 2], "a": ["x", "y"], "b": ["1", "2"]})

    def decode_feature_value(self, feature_col, encoded_value):
        return f"decoded-{feature_col}-{encoded_value}"


class DummyDemand:
    decision_log = [
        {"row_idx": 0, "col": 0, "action": 1, "result_value": 10},
        {"row_idx": 1, "col": 1, "action": 3, "result_value": 20},
    ]


def test_executor_uses_uniclean_for_repair(tmp_path):
    source = CleanedValueSource.from_df(
        pd.DataFrame({"index": [1, 2], "a": ["uni-x", "y"], "b": ["1", "2"]}),
        None,
        source_name="cached_uniclean",
    )
    result = execute_final_cleaning(
        DummyEncoded(),
        DummyDemand(),
        source,
        tmp_path,
    )
    assert result.cleaned_df.loc[0, "a"] == "uni-x"
    assert result.cleaned_df.loc[1, "b"] == "2"
    assert result.fallback_count == 0


def test_executor_can_keep_policy_value_estimator_for_ve(tmp_path):
    source = CleanedValueSource.from_df(
        pd.DataFrame({"index": [1, 2], "a": ["uni-x", "y"], "b": ["1", "2"]}),
        None,
        source_name="cached_uniclean",
    )
    result = execute_final_cleaning(
        DummyEncoded(),
        DummyDemand(),
        source,
        tmp_path,
        ve_source="policy",
    )
    assert result.cleaned_df.loc[0, "a"] == "uni-x"
    assert result.cleaned_df.loc[1, "b"] == "decoded-b-20"
    assert result.fallback_count == 0


def test_executor_requires_cached_uniclean_value(tmp_path):
    source = CleanedValueSource.from_df(
        pd.DataFrame({"index": [2], "a": ["uni-y"], "b": ["2"]}),
        None,
        source_name="cached_uniclean",
    )
    try:
        execute_final_cleaning(DummyEncoded(), DummyDemand(), source, tmp_path)
    except ValueError as exc:
        assert "Missing UniClean execution value" in str(exc)
    else:
        raise AssertionError("Expected missing cached value to fail loudly")
