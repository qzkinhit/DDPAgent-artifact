import json

import pandas as pd

from ads_clean.demo_data import action_summary, build_workflow_graph, dataset_catalog, load_run, runs_for_config
from ads_clean.executor import execute_final_cleaning
from ads_clean.repair_sources import CleanedValueSource


class TraceConfig:
    index_col = "index"


class TraceEncoded:
    feature_cols = ["a", "b"]
    config = TraceConfig()

    def __init__(self):
        self.dirty_df = pd.DataFrame({"index": [10, 11], "a": ["x", "y"], "b": ["1", "2"]})

    def decode_feature_value(self, feature_col, encoded_value):
        return f"decoded-{feature_col}-{encoded_value}"


class TraceDemand:
    decision_log = [
        {"row_idx": 0, "col": 0, "action": 1, "result_value": 10},
        {"row_idx": 1, "col": 1, "action": 3, "result_value": 20},
    ]


def test_executor_writes_real_operation_trace(tmp_path):
    source = CleanedValueSource.from_df(
        pd.DataFrame({"index": [10, 11], "a": ["uni-x", "y"], "b": ["1", "2"]}),
        None,
        source_name="cached_uniclean",
    )
    result = execute_final_cleaning(TraceEncoded(), TraceDemand(), source, tmp_path)
    assert result.operation_trace[0]["row_idx"] == 10
    assert result.operation_trace[0]["column"] == "a"
    assert result.operation_trace[0]["old_value"] == "x"
    assert result.operation_trace[0]["new_value"] == "uni-x"
    assert result.operation_trace[0]["changed"] is True


def test_demo_bundle_uses_only_existing_trace_files(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        json.dumps({
            "dataset": "beers",
            "scenario": "original",
            "task_type": "classification",
            "model_type": "random_forest",
            "rows": 2,
            "verifier_selected": "candidate",
        }),
        encoding="utf-8",
    )
    pd.DataFrame([
        {"row_idx": 0, "col": 0, "action": 0},
        {"row_idx": 1, "col": 1, "action": 1},
    ]).to_csv(run_dir / "action_trace.csv", index=False)
    pd.DataFrame([
        {"operation_id": 0, "operation_type": "cell_update", "changed": True}
    ]).to_csv(run_dir / "operation_trace.csv", index=False)

    bundle = load_run(run_dir)
    assert bundle["capabilities"]["action_trace"] is True
    assert bundle["capabilities"]["operator_trace"] is False

    graph = build_workflow_graph(bundle)
    node_ids = {node["id"] for node in graph["nodes"]}
    assert "controller" in node_ids
    assert "operators" not in node_ids

    summary = action_summary(bundle["action_trace"])
    assert set(summary["action_name"]) == {"no_op", "repair_value"}


def test_dataset_catalog_and_run_filter_are_configuration_driven(tmp_path):
    catalog = dataset_catalog()
    names = {row["name"] for row in catalog}
    assert {"beers", "flights", "hospitals", "rayyan", "tax"} <= names
    tax = next(row for row in catalog if row["name"] == "tax")
    assert tax["task_type"] == "regression"
    assert "ridge" in tax["candidate_models"]

    class Run:
        def __init__(self, dataset, scenario, model_type, error_rate=None):
            self.metrics = {
                "dataset": dataset,
                "scenario": scenario,
                "model_type": model_type,
                "error_rate": error_rate,
            }
            self.run_dir = tmp_path

    runs = [
        Run("beers", "original", "random_forest"),
        Run("beers", "original", "svm"),
        Run("tax", "original", "ridge"),
    ]
    matched = runs_for_config(runs, "beers", scenario="original", model_type="svm")
    assert len(matched) == 1
    assert matched[0].metrics["model_type"] == "svm"
