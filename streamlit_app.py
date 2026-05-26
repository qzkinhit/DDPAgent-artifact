from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ads_clean.demo_data import (
    action_records,
    action_summary,
    available_runs,
    build_workflow_graph,
    dataset_catalog,
    load_run,
    operation_summary,
    runs_for_config,
)


ROOT = Path(__file__).resolve().parent
ERROR_RATES = ["1", "025", "05", "075", "125", "15", "175", "2"]


TEXT: Dict[str, Dict[str, str]] = {
    "en": {
        "toggle": "中文",
        "title": "DDPAgent Console",
        "subtitle": "Demand-driven data governance agent",
        "caption": "Configure a downstream task, choose candidate models, inspect cached real artifacts, or launch a real traced execution.",
        "step1": "1. Data",
        "step2": "2. Task",
        "step3": "3. Candidates",
        "step4": "4. Execution",
        "dataset": "Dataset",
        "rows": "Rows",
        "target": "Target",
        "task": "Task",
        "scenario": "Scenario",
        "error_rate": "Injected error rate",
        "candidate_models": "Candidate models",
        "active_model": "Active model",
        "action_space": "Action space",
        "operators": "Operator profile",
        "cached_mode": "Use cached real artifact",
        "real_mode": "Run real pipeline",
        "no_cached": "No cached artifact matches this configuration. Use a real run to create one.",
        "selected_run": "Selected artifact",
        "run_dir": "Run directory",
        "trace_badge": "Trace status",
        "real_panel": "Execution settings",
        "real_note": "The button below runs the actual CLI with `--force-uniclean-run --trace-operators`. It does not simulate results.",
        "episodes": "Episodes",
        "max_errors": "Max detected errors",
        "single_max": "UniClean block threshold",
        "start_real": "Run active model",
        "running": "Running the real pipeline. This can take minutes because UniClean is executed instead of reading cached cleaned tables.",
        "finished": "Run completed. The new artifact is now available.",
        "failed": "Run failed.",
        "summary": "Run Summary",
        "fixed_delta": "Fixed split delta",
        "verifier_selected": "Verifier selected",
        "tabs_overview": "Overview",
        "tabs_workflow": "Workflow",
        "tabs_actions": "Actions",
        "tabs_operators": "Operators",
        "tabs_operations": "Data Operations",
        "tabs_verifier": "Verifier",
        "missing_operator": "This artifact has no runtime operator trace. DDPAgent will not display operator coverage or weights for this run.",
        "node_inspect": "Inspect node",
        "action_filter": "Action",
        "phase_filter": "Phase",
        "operator_filter": "Operator",
        "accepted_only": "Accepted only",
        "changed_only": "Changed only",
        "records": "Records",
        "changed": "Changed",
        "accepted": "Accepted",
        "before": "Before",
        "after": "After",
        "delta": "Delta",
        "no_rows": "No rows to display.",
        "refresh": "Refresh artifacts",
        "language": "Language",
    },
    "zh": {
        "toggle": "English",
        "title": "DDPAgent 控制台",
        "subtitle": "按需数据治理 Agent",
        "caption": "配置下游任务，选择候选模型，查看真实缓存 artifact，或启动带 trace 的真实执行。",
        "step1": "1. 数据",
        "step2": "2. 任务",
        "step3": "3. 候选",
        "step4": "4. 执行",
        "dataset": "数据集",
        "rows": "行数",
        "target": "目标列",
        "task": "任务",
        "scenario": "场景",
        "error_rate": "注入错误率",
        "candidate_models": "候选模型",
        "active_model": "当前模型",
        "action_space": "动作空间",
        "operators": "算子配置",
        "cached_mode": "使用真实缓存 artifact",
        "real_mode": "真实运行流程",
        "no_cached": "当前配置没有匹配的缓存 artifact。可以真实运行生成一个。",
        "selected_run": "选中的 artifact",
        "run_dir": "运行目录",
        "trace_badge": "Trace 状态",
        "real_panel": "执行设置",
        "real_note": "下方按钮会真实调用 CLI，并启用 `--force-uniclean-run --trace-operators`。不会模拟结果。",
        "episodes": "训练轮数",
        "max_errors": "最大检测错误数",
        "single_max": "UniClean 分块阈值",
        "start_real": "运行当前模型",
        "running": "正在真实运行流程。由于会执行 UniClean 而不是读取缓存，可能需要数分钟。",
        "finished": "运行完成。新的 artifact 已可选择。",
        "failed": "运行失败。",
        "summary": "运行摘要",
        "fixed_delta": "固定划分提升",
        "verifier_selected": "验证器选择",
        "tabs_overview": "总览",
        "tabs_workflow": "工作流",
        "tabs_actions": "动作",
        "tabs_operators": "算子",
        "tabs_operations": "数据操作",
        "tabs_verifier": "验证器",
        "missing_operator": "该 artifact 没有运行时算子 trace。DDPAgent 不会为该运行展示算子覆盖或权重。",
        "node_inspect": "查看节点",
        "action_filter": "动作",
        "phase_filter": "阶段",
        "operator_filter": "算子",
        "accepted_only": "只看验证器接受",
        "changed_only": "只看发生变化",
        "records": "记录数",
        "changed": "发生变化",
        "accepted": "已接受",
        "before": "之前",
        "after": "之后",
        "delta": "变化",
        "no_rows": "没有可展示记录。",
        "refresh": "刷新 artifact",
        "language": "语言",
    },
}


def main() -> None:
    st.set_page_config(page_title="DDPAgent Console", layout="wide")
    _inject_css()
    lang = _language()
    t = lambda key: TEXT[lang][key]

    with st.sidebar:
        st.button(t("toggle"), on_click=_toggle_language, use_container_width=True)
        if st.button(t("refresh"), use_container_width=True):
            st.rerun()

    st.markdown(f"<div class='hero'><div><div class='eyebrow'>{t('subtitle')}</div><h1>{t('title')}</h1><p>{t('caption')}</p></div></div>", unsafe_allow_html=True)

    catalog = dataset_catalog()
    runs = available_runs()
    config = _render_agent_setup(catalog, runs, lang)
    bundle = _render_execution_selector(runs, config, lang)

    if bundle is None:
        return

    _render_artifact_workspace(bundle, lang)


def _render_agent_setup(catalog: List[Dict[str, object]], runs, lang: str) -> Dict[str, object]:
    t = lambda key: TEXT[lang][key]
    names = [row["name"] for row in catalog]

    st.markdown("<div class='section-title'>Agent Setup</div>", unsafe_allow_html=True)
    cols = st.columns([1.1, 1.1, 1.2, 1.2])

    with cols[0]:
        box = st.container(border=True)
        box.markdown(f"<div class='step-label'>{t('step1')}</div>", unsafe_allow_html=True)
        dataset = box.selectbox(t("dataset"), names, key="dataset_select")
        meta = next(row for row in catalog if row["name"] == dataset)
        box.metric(t("rows"), f"{int(meta['rows']):,}")

    with cols[1]:
        box = st.container(border=True)
        box.markdown(f"<div class='step-label'>{t('step2')}</div>", unsafe_allow_html=True)
        scenario = box.radio(t("scenario"), ["original", "artificial"], horizontal=True, key=f"scenario_{dataset}")
        error_rate = box.selectbox(t("error_rate"), ERROR_RATES, key=f"rate_{dataset}") if scenario == "artificial" else None
        box.metric(t("task"), str(meta["task_type"]))
        box.caption(f"{t('target')}: {meta['target']}")

    with cols[2]:
        box = st.container(border=True)
        box.markdown(f"<div class='step-label'>{t('step3')}</div>", unsafe_allow_html=True)
        candidates = list(meta["candidate_models"])
        default_models = [meta["default_model"]] if meta["default_model"] in candidates else candidates[:1]
        selected_models = box.multiselect(t("candidate_models"), candidates, default=default_models, key=f"models_{dataset}")
        selected_models = selected_models or default_models
        active_model = box.radio(t("active_model"), selected_models, horizontal=True, key=f"active_model_{dataset}")
        box.caption(f"{t('action_space')}: no-op, repair, delete, replace")

    with cols[3]:
        box = st.container(border=True)
        box.markdown(f"<div class='step-label'>{t('step4')}</div>", unsafe_allow_html=True)
        matching = runs_for_config(runs, dataset, scenario=scenario, model_type=active_model, error_rate=error_rate)
        mode_options = [t("cached_mode"), t("real_mode")] if matching else [t("real_mode")]
        mode = box.radio("Mode", mode_options, horizontal=False, label_visibility="collapsed")
        box.metric(t("records"), len(matching))
        box.caption(f"{t('operators')}: {meta['cleaner_profile']}")

    return {
        "dataset": dataset,
        "scenario": scenario,
        "error_rate": error_rate,
        "candidate_models": selected_models,
        "model_type": active_model,
        "mode": "real" if mode == t("real_mode") else "cached",
        "matching_runs": matching,
        "meta": meta,
    }


def _render_execution_selector(runs, config: Dict[str, object], lang: str) -> Optional[Dict[str, object]]:
    t = lambda key: TEXT[lang][key]
    left, right = st.columns([1.25, 1])

    with left:
        st.markdown(f"<div class='panel-title'>{t('selected_run')}</div>", unsafe_allow_html=True)
        matching = config["matching_runs"]
        if config["mode"] == "cached" and matching:
            labels = [_artifact_label(run) for run in matching]
            selected_label = st.selectbox(t("selected_run"), labels, label_visibility="collapsed")
            selected = matching[labels.index(selected_label)]
            bundle = load_run(selected.run_dir)
            st.code(str(selected.run_dir))
            _render_trace_badges(bundle, lang)
            return bundle
        if config["mode"] == "cached":
            st.info(t("no_cached"))

    with right:
        _render_real_run_panel(config, lang)
    return None


def _render_real_run_panel(config: Dict[str, object], lang: str) -> None:
    t = lambda key: TEXT[lang][key]
    st.markdown(f"<div class='panel-title'>{t('real_panel')}</div>", unsafe_allow_html=True)
    st.caption(t("real_note"))
    with st.form("real_run_form"):
        cols = st.columns(3)
        episodes = int(cols[0].number_input(t("episodes"), min_value=1, max_value=1000, value=50, step=10))
        max_errors = int(cols[1].number_input(t("max_errors"), min_value=0, max_value=100000, value=50, step=10))
        single_max = int(cols[2].number_input(t("single_max"), min_value=100, max_value=100000, value=10000, step=1000))
        submitted = st.form_submit_button(t("start_real"), use_container_width=True)

    if submitted:
        with st.spinner(t("running")):
            result = _run_real_pipeline(
                dataset=str(config["dataset"]),
                scenario=str(config["scenario"]),
                error_rate=config["error_rate"],
                model=str(config["model_type"]),
                episodes=episodes,
                max_errors=max_errors,
                single_max=single_max,
            )
        if result.returncode == 0:
            st.success(t("finished"))
            st.code(result.stdout[-4000:])
            st.rerun()
        else:
            st.error(t("failed"))
            st.code((result.stdout + "\n" + result.stderr)[-8000:])


def _render_artifact_workspace(bundle: Dict[str, object], lang: str) -> None:
    t = lambda key: TEXT[lang][key]
    metrics = bundle["metrics"]
    caps = bundle["capabilities"]

    st.markdown(f"<div class='section-title'>{t('summary')}</div>", unsafe_allow_html=True)
    cols = st.columns(5)
    cols[0].metric(t("dataset"), str(metrics.get("dataset", "")))
    cols[1].metric(t("scenario"), f"{metrics.get('scenario', '')}/{metrics.get('error_rate') or 'native'}")
    cols[2].metric(t("model"), str(metrics.get("model_type", "")))
    cols[3].metric(t("fixed_delta"), _fmt_metric(metrics.get("downstream_fixed_delta", metrics.get("downstream_delta", ""))))
    cols[4].metric(t("verifier_selected"), str(metrics.get("verifier_selected", "")))

    if not caps["operator_trace"]:
        st.warning(t("missing_operator"))

    tab_labels = [
        t("tabs_overview"),
        t("tabs_workflow"),
        t("tabs_actions"),
        t("tabs_operators"),
        t("tabs_operations"),
        t("tabs_verifier"),
    ]
    overview, workflow, actions, operators, operations, verifier = st.tabs(tab_labels)

    with overview:
        _render_overview(bundle, lang)
    with workflow:
        _render_workflow(bundle, lang)
    with actions:
        _render_actions(bundle, lang)
    with operators:
        _render_operators(bundle, lang)
    with operations:
        _render_operations(bundle, lang)
    with verifier:
        _render_verifier(bundle, lang)


def _render_overview(bundle: Dict[str, object], lang: str) -> None:
    metrics = bundle["metrics"]
    trace = bundle["workflow"]
    cols = st.columns(2)
    with cols[0]:
        st.dataframe(pd.DataFrame([{
            "dataset": metrics.get("dataset"),
            "task_type": metrics.get("task_type"),
            "target": metrics.get("target"),
            "model_type": metrics.get("model_type"),
            "rows": metrics.get("rows"),
            "feature_count": metrics.get("feature_count"),
            "uniclean_cached": metrics.get("uniclean_cached"),
        }]), use_container_width=True)
    with cols[1]:
        st.json(trace.get("capabilities", bundle["capabilities"]))


def _render_workflow(bundle: Dict[str, object], lang: str) -> None:
    t = lambda key: TEXT[lang][key]
    graph = _localize_graph(build_workflow_graph(bundle), lang)
    st.plotly_chart(_workflow_figure(graph), use_container_width=True)
    labels = [f"{node['id']} | {node['label'].replace(chr(10), ' ')}" for node in graph["nodes"]]
    chosen = st.selectbox(t("node_inspect"), labels)
    node_id = chosen.split(" | ", 1)[0]
    node = next(node for node in graph["nodes"] if node["id"] == node_id)
    st.json(node)


def _render_actions(bundle: Dict[str, object], lang: str) -> None:
    t = lambda key: TEXT[lang][key]
    action_trace = bundle["action_trace"]
    summary = action_summary(action_trace)
    if summary.empty:
        st.info(t("no_rows"))
        return
    cols = st.columns([1, 1])
    with cols[0]:
        st.dataframe(summary, use_container_width=True)
    with cols[1]:
        fig = go.Figure(go.Bar(x=summary["action_name"], y=summary["count"], marker_color="#2563EB"))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), yaxis_title=t("records"))
        st.plotly_chart(fig, use_container_width=True)

    selected_action = st.selectbox(t("action_filter"), summary["action_name"].tolist())
    st.dataframe(action_records(action_trace, selected_action).head(500), use_container_width=True)


def _render_operators(bundle: Dict[str, object], lang: str) -> None:
    t = lambda key: TEXT[lang][key]
    operators = bundle["operator_trace"]
    if operators.empty:
        st.info(t("missing_operator"))
        return

    phase_options = ["all"] + sorted(operators["phase"].dropna().astype(str).unique().tolist()) if "phase" in operators.columns else ["all"]
    phase = st.selectbox(t("phase_filter"), phase_options)
    filtered = operators if phase == "all" else operators[operators["phase"].astype(str) == phase]
    st.dataframe(filtered, use_container_width=True)

    if "operator_id" in filtered.columns and not filtered.empty:
        operator_id = st.selectbox(t("operator_filter"), filtered["operator_id"].astype(str).drop_duplicates().tolist())
        selected = filtered[filtered["operator_id"].astype(str) == operator_id]
        st.dataframe(selected, use_container_width=True)

        rules = bundle["operation_rule_trace"]
        if not rules.empty and "operator_id" in rules.columns:
            block_ids = {operator_id}
            if not selected.empty and "node" in selected.columns:
                block_ids.add(f"block:{selected.iloc[0].get('node', '')}")
            st.dataframe(rules[rules["operator_id"].astype(str).isin(block_ids)].head(500), use_container_width=True)

        weights = bundle["operator_weight_trace"]
        if not weights.empty and "operator_id" in weights.columns:
            st.dataframe(weights[weights["operator_id"].astype(str) == operator_id], use_container_width=True)


def _render_operations(bundle: Dict[str, object], lang: str) -> None:
    t = lambda key: TEXT[lang][key]
    operations = bundle["operation_trace"]
    if operations.empty:
        st.info(t("no_rows"))
        return

    cols = st.columns(3)
    accepted_only = cols[0].checkbox(t("accepted_only"), value=False)
    changed_only = cols[1].checkbox(t("changed_only"), value=False)
    actions = ["all"] + sorted(operations["action"].dropna().astype(str).unique().tolist()) if "action" in operations.columns else ["all"]
    selected_action = cols[2].selectbox(t("action_filter"), actions)

    filtered = operations.copy()
    if accepted_only and "accepted_by_verifier" in filtered.columns:
        filtered = filtered[filtered["accepted_by_verifier"].fillna(False).astype(bool)]
    if changed_only and "changed" in filtered.columns:
        filtered = filtered[filtered["changed"].fillna(False).astype(bool)]
    if selected_action != "all" and "action" in filtered.columns:
        filtered = filtered[filtered["action"].astype(str) == selected_action]

    summary = operation_summary(filtered)
    if not summary.empty:
        st.dataframe(summary, use_container_width=True)
    st.dataframe(filtered.head(800), use_container_width=True)


def _render_verifier(bundle: Dict[str, object], lang: str) -> None:
    t = lambda key: TEXT[lang][key]
    metrics = bundle["metrics"]
    values = [
        ("fixed", metrics.get("downstream_fixed_before"), metrics.get("downstream_fixed_after")),
        ("random", metrics.get("downstream_before"), metrics.get("downstream_after")),
    ]
    rows = []
    for split, before, after in values:
        if before is None or after is None:
            continue
        rows.append({"split": split, t("before"): before, t("after"): after, t("delta"): float(after) - float(before)})
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(name=t("before"), x=df["split"], y=df[t("before")], marker_color="#94A3B8"))
        fig.add_trace(go.Bar(name=t("after"), x=df["split"], y=df[t("after")], marker_color="#2563EB"))
        fig.update_layout(barmode="group", height=330, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.json(metrics)


def _render_trace_badges(bundle: Dict[str, object], lang: str) -> None:
    t = lambda key: TEXT[lang][key]
    caps = bundle["capabilities"]
    good = [name for name, ok in caps.items() if ok]
    missing = [name for name, ok in caps.items() if not ok]
    st.caption(t("trace_badge"))
    st.markdown(" ".join(f"<span class='badge good'>{name}</span>" for name in good), unsafe_allow_html=True)
    if missing:
        st.markdown(" ".join(f"<span class='badge muted'>{name}</span>" for name in missing), unsafe_allow_html=True)


def _run_real_pipeline(dataset: str, scenario: str, error_rate: Optional[str], model: str, episodes: int, max_errors: int, single_max: int):
    cmd = [
        sys.executable,
        "-m",
        "ads_clean.cli",
        "run",
        "--dataset",
        dataset,
        "--scenario",
        scenario,
        "--output-root",
        "outputs/demo_trace_runs",
        "--profile",
        "default",
        "--episodes",
        str(episodes),
        "--rf-estimators",
        "10",
        "--base-cv-folds",
        "3",
        "--max-detected-errors",
        str(max_errors),
        "--single-max",
        str(single_max),
        "--model-type-override",
        model,
        "--force-uniclean-run",
        "--trace-operators",
        "--quiet",
    ]
    if scenario == "artificial":
        cmd.extend(["--error-rate", str(error_rate or "1"), "--result-assets"])
    env = os.environ.copy()
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, timeout=None)


def _artifact_label(run) -> str:
    metrics = run.metrics
    trace_flag = "trace" if (run.run_dir / "operator_trace.csv").exists() else "cached"
    delta = metrics.get("downstream_fixed_delta", metrics.get("downstream_delta", ""))
    try:
        delta_text = f"{float(delta):+.4f}"
    except (TypeError, ValueError):
        delta_text = "n/a"
    return f"{run.run_dir.name} | {trace_flag} | delta {delta_text}"


def _language() -> str:
    if "lang" not in st.session_state:
        st.session_state.lang = "en"
    return st.session_state.lang


def _toggle_language() -> None:
    st.session_state.lang = "zh" if st.session_state.get("lang", "en") == "en" else "en"


def _localize_graph(graph, lang: str):
    if lang == "en":
        return graph
    localized = {"nodes": [], "edges": []}
    for node in graph["nodes"]:
        new_node = dict(node)
        count = node.get("count", 0)
        if node["id"] == "controller":
            new_node["label"] = f"动作分配\n{count} 个错误"
        elif node["id"].startswith("action:"):
            action = node["id"].split(":", 1)[1]
            new_node["label"] = f"{_action_zh(action)}\n{count} 个单元"
        elif node["id"] == "operators":
            new_node["label"] = f"可用算子编排\n{count} 个算子"
        elif node["id"] == "operations":
            new_node["label"] = node["label"].replace("Data operation records", "数据操作记录").replace("accepted", "已接受")
        elif node["id"] == "verifier":
            new_node["label"] = node["label"].replace("Verifier", "验证器")
        localized["nodes"].append(new_node)
    edge_map = {
        "task and budget": "任务与预算",
        "runtime value source": "运行时值来源",
        "execute": "执行",
        "evaluate": "评估",
        "feedback": "反馈",
        "repair action": "修复动作",
        "value source": "值来源",
        "compiled values": "编译后的值",
    }
    for edge in graph["edges"]:
        new_edge = dict(edge)
        new_edge["label"] = edge_map.get(edge.get("label"), edge.get("label"))
        localized["edges"].append(new_edge)
    return localized


def _action_zh(action: str) -> str:
    return {
        "no_op": "不操作",
        "repair_value": "真值修复",
        "delete": "删除",
        "replace_nearby": "近邻替换",
    }.get(action, action)


def _fmt_metric(value) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def _workflow_figure(graph) -> go.Figure:
    nodes = graph["nodes"]
    edges = graph["edges"]
    layers = {
        "task": 0,
        "controller": 1,
        "action": 2,
        "operator_stage": 3,
        "operation": 4,
        "verifier": 5,
    }
    by_layer = {}
    for node in nodes:
        by_layer.setdefault(layers.get(node["kind"], 0), []).append(node)

    positions = {}
    for layer, layer_nodes in by_layer.items():
        total = len(layer_nodes)
        for i, node in enumerate(layer_nodes):
            positions[node["id"]] = (layer, (total - 1) / 2 - i)

    edge_x = []
    edge_y = []
    annotations = []
    for edge in edges:
        if edge["source"] not in positions or edge["target"] not in positions:
            continue
        x0, y0 = positions[edge["source"]]
        x1, y1 = positions[edge["target"]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        annotations.append(
            dict(
                x=x1,
                y=y1,
                ax=x0,
                ay=y0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.2,
                arrowcolor="#64748B",
                opacity=0.8,
            )
        )

    color_map = {
        "task": "#2563EB",
        "controller": "#7C3AED",
        "action": "#059669",
        "operator_stage": "#D97706",
        "operation": "#DC2626",
        "verifier": "#475569",
    }
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1.4, color="#CBD5E1"), hoverinfo="none"))
    fig.add_trace(
        go.Scatter(
            x=[positions[node["id"]][0] for node in nodes],
            y=[positions[node["id"]][1] for node in nodes],
            mode="markers+text",
            marker=dict(size=50, color=[color_map.get(node["kind"], "#475569") for node in nodes], line=dict(width=1.8, color="white")),
            text=[node["label"] for node in nodes],
            textposition="bottom center",
            textfont=dict(size=13, color="#0F172A"),
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        height=460,
        margin=dict(l=20, r=20, t=30, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        xaxis=dict(visible=False, range=(-0.5, 5.5)),
        yaxis=dict(visible=False),
        annotations=annotations,
    )
    return fig


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        .hero {
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 22px 26px;
            background: #FFFFFF;
            margin-bottom: 18px;
        }
        .hero h1 {
            margin: 2px 0 4px 0;
            color: #0F172A;
            font-size: 34px;
            letter-spacing: 0;
        }
        .hero p { margin: 0; color: #475569; font-size: 15px; }
        .eyebrow {
            color: #2563EB;
            font-weight: 700;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0;
        }
        .section-title {
            margin: 18px 0 10px 0;
            font-size: 18px;
            font-weight: 700;
            color: #0F172A;
        }
        .panel-title {
            margin: 4px 0 8px 0;
            font-size: 15px;
            font-weight: 700;
            color: #0F172A;
        }
        .step-card {
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 14px 14px 10px 14px;
            background: #FFFFFF;
            min-height: 235px;
        }
        .step-label {
            font-size: 13px;
            font-weight: 700;
            color: #334155;
            margin-bottom: 8px;
        }
        .badge {
            display: inline-block;
            border-radius: 999px;
            padding: 3px 9px;
            margin: 3px 4px 3px 0;
            font-size: 12px;
            border: 1px solid #CBD5E1;
        }
        .badge.good { background: #ECFDF5; color: #047857; border-color: #A7F3D0; }
        .badge.muted { background: #F8FAFC; color: #64748B; }
        div[data-testid="stMetricValue"] { font-size: 22px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
