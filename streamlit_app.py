from __future__ import annotations

import os
import subprocess
import sys
from html import escape
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demandprep.demo_data import (
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


ICON_PATHS = {
    "agent": "<path d='M12 2v4'/><path d='M12 18v4'/><path d='M4.93 4.93l2.83 2.83'/><path d='M16.24 16.24l2.83 2.83'/><path d='M2 12h4'/><path d='M18 12h4'/><path d='M4.93 19.07l2.83-2.83'/><path d='M16.24 7.76l2.83-2.83'/><circle cx='12' cy='12' r='4'/>",
    "database": "<ellipse cx='12' cy='5' rx='8' ry='3'/><path d='M4 5v6c0 1.66 3.58 3 8 3s8-1.34 8-3V5'/><path d='M4 11v6c0 1.66 3.58 3 8 3s8-1.34 8-3v-6'/>",
    "target": "<circle cx='12' cy='12' r='9'/><circle cx='12' cy='12' r='5'/><circle cx='12' cy='12' r='1'/>",
    "cpu": "<rect x='7' y='7' width='10' height='10' rx='2'/><path d='M9 1v3'/><path d='M15 1v3'/><path d='M9 20v3'/><path d='M15 20v3'/><path d='M20 9h3'/><path d='M20 15h3'/><path d='M1 9h3'/><path d='M1 15h3'/>",
    "play": "<path d='M5 3l14 9-14 9V3z'/>",
    "archive": "<rect x='3' y='4' width='18' height='4' rx='1'/><path d='M5 8v11a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V8'/><path d='M10 12h4'/>",
    "workflow": "<rect x='3' y='3' width='6' height='6' rx='1'/><rect x='15' y='3' width='6' height='6' rx='1'/><rect x='9' y='15' width='6' height='6' rx='1'/><path d='M9 6h6'/><path d='M12 9v6'/>",
    "activity": "<path d='M22 12h-4l-3 8L9 4l-3 8H2'/>",
    "wrench": "<path d='M14.7 6.3a4 4 0 0 0-5 5L3 18l3 3 6.7-6.7a4 4 0 0 0 5-5l-2.4 2.4-2.6-2.6 2.4-2.4z'/>",
    "edit": "<path d='M12 20h9'/><path d='M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z'/>",
    "shield": "<path d='M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z'/>",
    "check": "<path d='M20 6 9 17l-5-5'/>",
    "alert": "<path d='M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z'/><path d='M12 9v4'/><path d='M12 17h.01'/>",
    "refresh": "<path d='M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16'/><path d='M3 21v-5h5'/><path d='M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8'/><path d='M21 3v5h-5'/>",
    "globe": "<circle cx='12' cy='12' r='10'/><path d='M2 12h20'/><path d='M12 2a15.3 15.3 0 0 1 0 20'/><path d='M12 2a15.3 15.3 0 0 0 0 20'/>",
    "table": "<path d='M3 5h18v14H3z'/><path d='M3 10h18'/><path d='M9 5v14'/><path d='M15 5v14'/>",
}


TEXT: Dict[str, Dict[str, str]] = {
    "en": {
        "toggle": "中文",
        "title": "DemandPrep Console",
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
        "model": "Model",
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
        "missing_operator": "This artifact has no runtime operator trace. DemandPrep will not display operator coverage or weights for this run.",
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
        "title": "DemandPrep 控制台",
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
        "model": "模型",
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
        "missing_operator": "该 artifact 没有运行时算子 trace。DemandPrep 不会为该运行展示算子覆盖或权重。",
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
    st.set_page_config(page_title="DemandPrep Console", layout="wide")
    _inject_css()
    lang = _language()
    t = lambda key: TEXT[lang][key]

    with st.sidebar:
        st.markdown(_sidebar_brand(lang), unsafe_allow_html=True)
        st.button(t("toggle"), on_click=_toggle_language, use_container_width=True)
        if st.button(t("refresh"), use_container_width=True):
            st.rerun()

    st.markdown(_hero_html(lang), unsafe_allow_html=True)

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

    st.markdown(_section_title("agent", "Agent Setup"), unsafe_allow_html=True)
    cols = st.columns([1.1, 1.1, 1.2, 1.2])

    with cols[0]:
        box = st.container(border=True)
        box.markdown(_step_heading("database", t("step1"), "Select the table artifact" if lang == "en" else "选择表格 artifact"), unsafe_allow_html=True)
        dataset = box.selectbox(t("dataset"), names, key="dataset_select")
        meta = next(row for row in catalog if row["name"] == dataset)
        box.markdown(_metric_card(t("rows"), f"{int(meta['rows']):,}", str(meta["source"]), "table", "blue"), unsafe_allow_html=True)

    with cols[1]:
        box = st.container(border=True)
        box.markdown(_step_heading("target", t("step2"), "Bind utility objective" if lang == "en" else "绑定下游目标"), unsafe_allow_html=True)
        scenario = box.radio(t("scenario"), ["original", "artificial"], horizontal=True, key=f"scenario_{dataset}")
        error_rate = box.selectbox(t("error_rate"), ERROR_RATES, key=f"rate_{dataset}") if scenario == "artificial" else None
        box.markdown(_metric_card(t("task"), str(meta["task_type"]), f"{t('target')}: {meta['target']}", "target", "green"), unsafe_allow_html=True)

    with cols[2]:
        box = st.container(border=True)
        box.markdown(_step_heading("cpu", t("step3"), "Choose downstream candidates" if lang == "en" else "选择候选下游模型"), unsafe_allow_html=True)
        candidates = list(meta["candidate_models"])
        default_models = [meta["default_model"]] if meta["default_model"] in candidates else candidates[:1]
        selected_models = box.multiselect(t("candidate_models"), candidates, default=default_models, key=f"models_{dataset}")
        selected_models = selected_models or default_models
        active_model = box.radio(t("active_model"), selected_models, horizontal=True, key=f"active_model_{dataset}")
        box.markdown(_chip_row([f"{t('action_space')}", "no-op", "repair", "delete", "replace"]), unsafe_allow_html=True)

    with cols[3]:
        box = st.container(border=True)
        box.markdown(_step_heading("play", t("step4"), "Inspect or execute" if lang == "en" else "查看或真实执行"), unsafe_allow_html=True)
        matching = runs_for_config(runs, dataset, scenario=scenario, model_type=active_model, error_rate=error_rate)
        mode_options = [t("cached_mode"), t("real_mode")] if matching else [t("real_mode")]
        mode = box.radio("Mode", mode_options, horizontal=False, label_visibility="collapsed")
        box.markdown(_metric_card(t("records"), len(matching), f"{t('operators')}: {meta['cleaner_profile']}", "archive", "amber"), unsafe_allow_html=True)

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
        st.markdown(_panel_title("archive", t("selected_run")), unsafe_allow_html=True)
        matching = config["matching_runs"]
        if config["mode"] == "cached" and matching:
            labels = [_artifact_label(run) for run in matching]
            selected_label = st.selectbox(t("selected_run"), labels, label_visibility="collapsed")
            selected = matching[labels.index(selected_label)]
            bundle = load_run(selected.run_dir)
            st.markdown(_artifact_card(selected, bundle, lang), unsafe_allow_html=True)
            _render_trace_badges(bundle, lang)
            return bundle
        if config["mode"] == "cached":
            st.info(t("no_cached"))

    with right:
        _render_real_run_panel(config, lang)
    return None


def _render_real_run_panel(config: Dict[str, object], lang: str) -> None:
    t = lambda key: TEXT[lang][key]
    st.markdown(_panel_title("play", t("real_panel")), unsafe_allow_html=True)
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

    st.markdown(_section_title("activity", t("summary")), unsafe_allow_html=True)
    cols = st.columns(5)
    cols[0].markdown(_metric_card(t("dataset"), str(metrics.get("dataset", "")), "artifact", "database", "blue"), unsafe_allow_html=True)
    cols[1].markdown(_metric_card(t("scenario"), f"{metrics.get('scenario', '')}/{metrics.get('error_rate') or 'native'}", "data setting", "target", "green"), unsafe_allow_html=True)
    cols[2].markdown(_metric_card(t("model"), str(metrics.get("model_type", "")), str(metrics.get("task_type", "")), "cpu", "purple"), unsafe_allow_html=True)
    cols[3].markdown(_metric_card(t("fixed_delta"), _fmt_metric(metrics.get("downstream_fixed_delta", metrics.get("downstream_delta", ""))), "utility", "activity", "amber"), unsafe_allow_html=True)
    cols[4].markdown(_metric_card(t("verifier_selected"), str(metrics.get("verifier_selected", "")), "policy", "shield", "slate"), unsafe_allow_html=True)

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
    t = lambda key: TEXT[lang][key]
    metrics = bundle["metrics"]
    trace = bundle["workflow"]
    cols = st.columns([1.1, 0.9])
    with cols[0]:
        st.markdown(_panel_title("database", "Artifact profile" if lang == "en" else "Artifact 概况"), unsafe_allow_html=True)
        st.markdown(_profile_grid({
            "dataset": metrics.get("dataset"),
            "task_type": metrics.get("task_type"),
            "target": metrics.get("target"),
            "model_type": metrics.get("model_type"),
            "rows": metrics.get("rows"),
            "feature_count": metrics.get("feature_count"),
            "uniclean_cached": metrics.get("uniclean_cached"),
        }), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(_panel_title("check", t("trace_badge")), unsafe_allow_html=True)
        capabilities = trace.get("capabilities", bundle["capabilities"])
        st.markdown(_capability_grid(capabilities), unsafe_allow_html=True)


def _render_workflow(bundle: Dict[str, object], lang: str) -> None:
    t = lambda key: TEXT[lang][key]
    graph = _localize_graph(build_workflow_graph(bundle), lang)
    st.plotly_chart(_workflow_figure(graph), use_container_width=True)
    labels = [f"{node['id']} | {node['label'].replace(chr(10), ' ')}" for node in graph["nodes"]]
    chosen = st.selectbox(t("node_inspect"), labels)
    node_id = chosen.split(" | ", 1)[0]
    node = next(node for node in graph["nodes"] if node["id"] == node_id)
    st.markdown(_node_card(node), unsafe_allow_html=True)


def _render_actions(bundle: Dict[str, object], lang: str) -> None:
    t = lambda key: TEXT[lang][key]
    action_trace = bundle["action_trace"]
    summary = action_summary(action_trace)
    if summary.empty:
        st.info(t("no_rows"))
        return
    st.markdown(_action_cards(summary), unsafe_allow_html=True)
    cols = st.columns([0.95, 1.05])
    with cols[0]:
        st.dataframe(summary, use_container_width=True, hide_index=True)
    with cols[1]:
        fig = go.Figure(go.Bar(
            x=summary["action_name"],
            y=summary["count"],
            marker=dict(color=["#2563EB", "#059669", "#D97706", "#DC2626"][: len(summary)]),
            text=summary["count"],
            textposition="outside",
        ))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=26, b=10), yaxis_title=t("records"), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    selected_action = st.selectbox(t("action_filter"), summary["action_name"].tolist())
    st.dataframe(action_records(action_trace, selected_action).head(500), use_container_width=True)


def _render_operators(bundle: Dict[str, object], lang: str) -> None:
    t = lambda key: TEXT[lang][key]
    operators = bundle["operator_trace"]
    if operators.empty:
        st.info(t("missing_operator"))
        return

    st.markdown(_metric_row([
        (t("records"), len(operators), "operator rows", "wrench", "blue"),
        ("rules", len(bundle["operation_rule_trace"]), "generated", "edit", "green"),
        ("weights", len(bundle["operator_weight_trace"]), "feedback", "activity", "amber"),
    ]), unsafe_allow_html=True)

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

    changed_total = int(operations["changed"].fillna(False).astype(bool).sum()) if "changed" in operations.columns else len(operations)
    accepted_total = int(operations["accepted_by_verifier"].fillna(False).astype(bool).sum()) if "accepted_by_verifier" in operations.columns else len(operations)
    st.markdown(_metric_row([
        (t("records"), len(operations), "operation rows", "edit", "blue"),
        (t("changed"), changed_total, "cell or row changes", "activity", "green"),
        (t("accepted"), accepted_total, "verifier", "shield", "slate"),
    ]), unsafe_allow_html=True)

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


def _icon(name: str, size: int = 18, color: str = "currentColor") -> str:
    body = ICON_PATHS.get(name, ICON_PATHS["agent"])
    return (
        f"<svg class='icon' width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' "
        f"stroke='{color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>{body}</svg>"
    )


def _sidebar_brand(lang: str) -> str:
    title = "DemandPrep"
    subtitle = "Data governance console" if lang == "en" else "数据治理控制台"
    return (
        "<div class='sidebar-brand'>"
        f"<div class='brand-icon'>{_icon('agent', 22)}</div>"
        f"<div><div class='brand-title'>{title}</div><div class='brand-subtitle'>{subtitle}</div></div>"
        "</div>"
    )


def _hero_html(lang: str) -> str:
    t = lambda key: TEXT[lang][key]
    chips = ["Action allocation", "Operator orchestration", "Verifier feedback"] if lang == "en" else ["动作分配", "算子编排", "验证反馈"]
    return (
        "<div class='hero'>"
        "<div class='hero-copy'>"
        f"<div class='eyebrow'>{_icon('agent', 15)}{escape(t('subtitle'))}</div>"
        f"<h1>{escape(t('title'))}</h1>"
        f"<p>{escape(t('caption'))}</p>"
        f"<div class='hero-chips'>{''.join(f'<span>{escape(chip)}</span>' for chip in chips)}</div>"
        "</div>"
        "<div class='hero-panel'>"
        f"<div class='hero-panel-icon'>{_icon('workflow', 30)}</div>"
        "<div class='hero-panel-title'>control loop</div>"
        "<div class='hero-panel-flow'>Task -> Policy -> Operators -> Verifier</div>"
        "</div>"
        "</div>"
    )


def _section_title(icon: str, text: str) -> str:
    return f"<div class='section-title'>{_icon(icon, 18)}<span>{escape(str(text))}</span></div>"


def _panel_title(icon: str, text: str) -> str:
    return f"<div class='panel-title'>{_icon(icon, 17)}<span>{escape(str(text))}</span></div>"


def _step_heading(icon: str, title: str, subtitle: str) -> str:
    return (
        "<div class='step-heading'>"
        f"<div class='step-icon'>{_icon(icon, 18)}</div>"
        f"<div><div class='step-title'>{escape(str(title))}</div><div class='step-subtitle'>{escape(str(subtitle))}</div></div>"
        "</div>"
    )


def _metric_card(title, value, caption: str, icon: str, tone: str = "blue") -> str:
    return (
        f"<div class='metric-card tone-{tone}'>"
        f"<div class='metric-icon'>{_icon(icon, 18)}</div>"
        f"<div><div class='metric-title'>{escape(str(title))}</div>"
        f"<div class='metric-value'>{escape(str(value))}</div>"
        f"<div class='metric-caption'>{escape(str(caption))}</div></div>"
        "</div>"
    )


def _metric_row(items) -> str:
    cards = "".join(_metric_card(title, value, caption, icon, tone) for title, value, caption, icon, tone in items)
    return f"<div class='metric-row'>{cards}</div>"


def _chip_row(chips: List[str]) -> str:
    if not chips:
        return ""
    first, rest = chips[0], chips[1:]
    return (
        "<div class='chip-row'>"
        f"<span class='chip-label'>{escape(str(first))}</span>"
        + "".join(f"<span class='chip'>{escape(str(chip))}</span>" for chip in rest)
        + "</div>"
    )


def _artifact_card(run, bundle: Dict[str, object], lang: str) -> str:
    metrics = bundle["metrics"]
    t = lambda key: TEXT[lang][key]
    trace = "runtime trace" if bundle["capabilities"].get("operator_trace") else "cached only"
    delta = _fmt_metric(metrics.get("downstream_fixed_delta", metrics.get("downstream_delta", "")))
    return (
        "<div class='artifact-card'>"
        f"<div class='artifact-icon'>{_icon('archive', 22)}</div>"
        "<div class='artifact-body'>"
        f"<div class='artifact-title'>{escape(str(run.run_dir.name))}</div>"
        f"<div class='artifact-meta'>{escape(str(run.run_dir))}</div>"
        "<div class='artifact-stats'>"
        f"<span>{escape(t('model'))}: {escape(str(metrics.get('model_type', '')))}</span>"
        f"<span>{escape(t('fixed_delta'))}: {escape(delta)}</span>"
        f"<span>{escape(trace)}</span>"
        "</div></div></div>"
    )


def _profile_grid(values: Dict[str, object]) -> str:
    rows = []
    for key, value in values.items():
        rows.append(
            "<div class='profile-item'>"
            f"<div class='profile-key'>{escape(str(key))}</div>"
            f"<div class='profile-value'>{escape(str(value))}</div>"
            "</div>"
        )
    return f"<div class='profile-grid'>{''.join(rows)}</div>"


def _capability_grid(capabilities: Dict[str, object]) -> str:
    cards = []
    for key, value in capabilities.items():
        cls = "capability-on" if bool(value) else "capability-off"
        icon = "check" if bool(value) else "alert"
        state = "ready" if bool(value) else "missing"
        cards.append(
            f"<div class='capability {cls}'>"
            f"{_icon(icon, 16)}<div><div class='capability-name'>{escape(str(key))}</div>"
            f"<div class='capability-state'>{state}</div></div></div>"
        )
    return f"<div class='capability-grid'>{''.join(cards)}</div>"


def _node_card(node: Dict[str, object]) -> str:
    kind = str(node.get("kind", "node"))
    icon = {
        "task": "database",
        "controller": "agent",
        "action": "activity",
        "operator_stage": "wrench",
        "operation": "edit",
        "verifier": "shield",
    }.get(kind, "workflow")
    return (
        "<div class='node-card'>"
        f"<div class='node-icon'>{_icon(icon, 22)}</div>"
        f"<div><div class='node-title'>{escape(str(node.get('label', ''))).replace(chr(10), '<br>')}</div>"
        f"<div class='node-meta'>{escape(kind)} · count {escape(str(node.get('count', '')))}</div></div>"
        "</div>"
    )


def _action_cards(summary: pd.DataFrame) -> str:
    total = int(summary["count"].sum()) if "count" in summary.columns else 0
    icon_map = {
        "no_op": "shield",
        "repair_value": "wrench",
        "delete": "alert",
        "replace_nearby": "edit",
    }
    tone_map = {
        "no_op": "slate",
        "repair_value": "green",
        "delete": "amber",
        "replace_nearby": "blue",
    }
    cards = []
    for _, row in summary.iterrows():
        action = str(row["action_name"])
        count = int(row["count"])
        pct = count / total * 100 if total else 0
        cards.append(
            f"<div class='action-card tone-{tone_map.get(action, 'blue')}'>"
            f"<div class='action-top'>{_icon(icon_map.get(action, 'activity'), 18)}<span>{escape(action)}</span></div>"
            f"<div class='action-count'>{count}</div>"
            f"<div class='action-bar'><span style='width:{pct:.1f}%'></span></div>"
            f"<div class='action-pct'>{pct:.1f}% of decisions</div>"
            "</div>"
        )
    return f"<div class='action-card-row'>{''.join(cards)}</div>"


def _run_real_pipeline(dataset: str, scenario: str, error_rate: Optional[str], model: str, episodes: int, max_errors: int, single_max: int):
    cmd = [
        sys.executable,
        "-m",
        "demandprep.cli",
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
        :root {
            --ink: #0F172A;
            --muted: #64748B;
            --line: #E2E8F0;
            --panel: #FFFFFF;
            --page: #F8FAFC;
            --blue: #2563EB;
            --green: #059669;
            --amber: #D97706;
            --purple: #7C3AED;
            --red: #DC2626;
            --slate: #475569;
        }
        .stApp { background: var(--page); }
        .block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1420px; }
        section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid var(--line); }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--line);
            border-radius: 14px;
            background: #FFFFFF;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.04);
        }
        .icon { vertical-align: -3px; flex: 0 0 auto; }
        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 4px 18px 4px;
            border-bottom: 1px solid var(--line);
            margin-bottom: 14px;
        }
        .brand-icon {
            width: 38px;
            height: 38px;
            border-radius: 12px;
            background: #EFF6FF;
            color: var(--blue);
            display: grid;
            place-items: center;
            border: 1px solid #BFDBFE;
        }
        .brand-title { font-size: 16px; font-weight: 800; color: var(--ink); line-height: 1.1; }
        .brand-subtitle { font-size: 12px; color: var(--muted); margin-top: 2px; }
        .hero {
            border: 1px solid #E2E8F0;
            border-radius: 18px;
            padding: 24px 28px;
            background: #FFFFFF;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 22px;
            box-shadow: 0 18px 42px rgba(15, 23, 42, 0.06);
            border-left: 5px solid var(--blue);
        }
        .hero h1 {
            margin: 5px 0 8px 0;
            color: #0F172A;
            font-size: 38px;
            line-height: 1.05;
            letter-spacing: 0;
        }
        .hero p { margin: 0; color: #475569; font-size: 15px; max-width: 780px; }
        .hero-copy { min-width: 0; }
        .hero-chips { margin-top: 16px; display: flex; flex-wrap: wrap; gap: 8px; }
        .hero-chips span {
            border: 1px solid #CBD5E1;
            border-radius: 999px;
            padding: 5px 10px;
            color: #334155;
            background: #F8FAFC;
            font-size: 12px;
            font-weight: 650;
        }
        .hero-panel {
            min-width: 260px;
            border: 1px solid #DBEAFE;
            border-radius: 16px;
            padding: 16px;
            background: #EFF6FF;
            color: #1E3A8A;
        }
        .hero-panel-icon {
            width: 46px;
            height: 46px;
            border-radius: 14px;
            display: grid;
            place-items: center;
            background: #FFFFFF;
            color: var(--blue);
            border: 1px solid #BFDBFE;
        }
        .hero-panel-title { margin-top: 12px; font-size: 13px; font-weight: 800; text-transform: uppercase; letter-spacing: 0; }
        .hero-panel-flow { margin-top: 5px; color: #334155; font-size: 13px; }
        .eyebrow {
            color: #2563EB;
            font-weight: 700;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0;
            display: flex;
            align-items: center;
            gap: 7px;
        }
        .section-title {
            margin: 20px 0 12px 0;
            font-size: 18px;
            font-weight: 700;
            color: #0F172A;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .panel-title {
            margin: 8px 0 10px 0;
            font-size: 15px;
            font-weight: 700;
            color: #0F172A;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .step-heading {
            display: flex;
            gap: 10px;
            align-items: center;
            margin-bottom: 8px;
        }
        .step-icon {
            width: 34px;
            height: 34px;
            border-radius: 11px;
            display: grid;
            place-items: center;
            background: #EFF6FF;
            color: var(--blue);
            border: 1px solid #BFDBFE;
        }
        .step-title {
            font-size: 13px;
            font-weight: 800;
            color: var(--ink);
        }
        .step-subtitle {
            color: var(--muted);
            font-size: 12px;
            margin-top: 1px;
        }
        .metric-card {
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 12px;
            background: #FFFFFF;
            display: flex;
            gap: 10px;
            align-items: flex-start;
            min-height: 86px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
        }
        .metric-card:hover, .artifact-card:hover, .action-card:hover, .node-card:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
        }
        .metric-card, .artifact-card, .action-card, .node-card { transition: transform 140ms ease, box-shadow 140ms ease; }
        .metric-icon, .artifact-icon, .node-icon {
            width: 34px;
            height: 34px;
            border-radius: 11px;
            display: grid;
            place-items: center;
            border: 1px solid currentColor;
        }
        .tone-blue .metric-icon, .tone-blue .action-top { color: var(--blue); }
        .tone-green .metric-icon, .tone-green .action-top { color: var(--green); }
        .tone-amber .metric-icon, .tone-amber .action-top { color: var(--amber); }
        .tone-purple .metric-icon, .tone-purple .action-top { color: var(--purple); }
        .tone-slate .metric-icon, .tone-slate .action-top { color: var(--slate); }
        .metric-title { font-size: 12px; color: var(--muted); font-weight: 650; }
        .metric-value { color: var(--ink); font-weight: 820; font-size: 21px; line-height: 1.15; margin-top: 3px; overflow-wrap: anywhere; }
        .metric-caption { color: var(--muted); font-size: 12px; margin-top: 4px; }
        .metric-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 14px;
        }
        .chip-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
        .chip-label, .chip {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 4px 8px;
            font-size: 12px;
            border: 1px solid var(--line);
        }
        .chip-label { color: var(--slate); background: #F8FAFC; font-weight: 700; }
        .chip { color: #075985; background: #F0F9FF; border-color: #BAE6FD; }
        .artifact-card {
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 14px;
            background: #FFFFFF;
            display: flex;
            gap: 12px;
            align-items: flex-start;
            margin-bottom: 8px;
        }
        .artifact-icon { color: var(--blue); background: #EFF6FF; border-color: #BFDBFE; }
        .artifact-body { min-width: 0; }
        .artifact-title { font-weight: 800; color: var(--ink); font-size: 15px; }
        .artifact-meta { color: var(--muted); font-size: 12px; margin-top: 3px; word-break: break-all; }
        .artifact-stats { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
        .artifact-stats span {
            border-radius: 999px;
            padding: 4px 8px;
            font-size: 12px;
            background: #F8FAFC;
            color: #334155;
            border: 1px solid var(--line);
        }
        .profile-grid, .capability-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
            margin-bottom: 14px;
        }
        .profile-item, .capability {
            border: 1px solid var(--line);
            border-radius: 12px;
            background: #FFFFFF;
            padding: 11px;
        }
        .profile-key, .capability-state { color: var(--muted); font-size: 12px; }
        .profile-value, .capability-name { color: var(--ink); font-size: 14px; font-weight: 760; margin-top: 3px; word-break: break-word; }
        .capability { display: flex; gap: 9px; align-items: center; }
        .capability-on { color: var(--green); background: #F0FDF4; border-color: #BBF7D0; }
        .capability-off { color: var(--amber); background: #FFFBEB; border-color: #FDE68A; }
        .node-card {
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 14px;
            background: #FFFFFF;
            display: flex;
            gap: 12px;
            align-items: center;
            margin-top: 8px;
        }
        .node-icon { color: var(--purple); background: #F5F3FF; border-color: #DDD6FE; }
        .node-title { color: var(--ink); font-weight: 820; font-size: 15px; }
        .node-meta { color: var(--muted); font-size: 12px; margin-top: 4px; }
        .action-card-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 14px;
        }
        .action-card {
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 13px;
            background: #FFFFFF;
        }
        .action-top { display: flex; align-items: center; gap: 8px; font-weight: 800; font-size: 13px; }
        .action-count { font-size: 26px; color: var(--ink); font-weight: 850; margin-top: 10px; }
        .action-bar { height: 7px; border-radius: 999px; background: #E2E8F0; overflow: hidden; margin-top: 8px; }
        .action-bar span { display: block; height: 100%; border-radius: 999px; background: currentColor; }
        .action-pct { color: var(--muted); font-size: 12px; margin-top: 6px; }
        .tone-blue .action-bar span { background: var(--blue); }
        .tone-green .action-bar span { background: var(--green); }
        .tone-amber .action-bar span { background: var(--amber); }
        .tone-slate .action-bar span { background: var(--slate); }
        div[data-testid="stButton"] > button {
            border-radius: 10px;
            border: 1px solid #BFDBFE;
            background: #EFF6FF;
            color: #1D4ED8;
            font-weight: 760;
        }
        div[data-testid="stFormSubmitButton"] > button {
            border-radius: 11px;
            border: 1px solid #1D4ED8;
            background: #2563EB;
            color: #FFFFFF;
            font-weight: 780;
            height: 42px;
        }
        div[data-testid="stTabs"] button p { font-weight: 750; font-size: 14px; }
        div[data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
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
        @media (max-width: 900px) {
            .hero { flex-direction: column; align-items: stretch; }
            .hero-panel { min-width: 0; }
            .metric-row, .action-card-row, .profile-grid, .capability-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
