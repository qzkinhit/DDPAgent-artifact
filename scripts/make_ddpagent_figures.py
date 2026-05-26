#!/usr/bin/env python3
"""Generate compact figures for the DDPAgent workshop draft."""

from pathlib import Path
import ast

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
OUT_FINAL = ROOT / "outputs" / "experiments_20260519_final"
OUT_VERIFIER = ROOT / "outputs" / "experiments_20260519_verifier"
OUT_HOSPITAL_TARGET = ROOT / "outputs" / "experiments_20260520_hospital_measurecode"


COLORS = {
    "ddp": "#C44E52",
    "blue": "#4C72B0",
    "orange": "#DD8452",
    "green": "#55A868",
    "purple": "#8172B3",
    "gray": "#7A7A7A",
    "light_gray": "#F3F4F6",
    "line": "#2F2F2F",
}


def setup():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 1.8,
            "xtick.major.width": 1.6,
            "ytick.major.width": 1.6,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "figure.dpi": 450,
            "savefig.dpi": 450,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "stix",
        }
    )


def rounded_box(ax, x, y, w, h, text, fc="white", ec=None, lw=1.2, fontsize=8.5):
    ec = ec or COLORS["line"]
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return box


def arrow(ax, start, end, color=None, rad=0.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=1.2,
        color=color or COLORS["line"],
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    return arr


def make_framework_placeholder():
    fig, ax = plt.subplots(figsize=(7.2, 2.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    blocks = [
        (0.03, 0.52, 0.17, 0.3, "Task Demand\nmetric, data profile,\noperator library", COLORS["light_gray"]),
        (0.25, 0.52, 0.2, 0.3, "Agentic Action\nAllocation", "#EAF2FB"),
        (0.51, 0.52, 0.22, 0.3, "Operator-Grounded\nExecution", "#EDF7ED"),
        (0.78, 0.52, 0.18, 0.3, "Prepared Data\n+ Trace", "#FFF3E6"),
    ]
    for x, y, w, h, text, fc in blocks:
        rounded_box(ax, x, y, w, h, text, fc=fc, fontsize=8.4)
    for start, end in [((0.20, 0.67), (0.25, 0.67)), ((0.45, 0.67), (0.51, 0.67)), ((0.73, 0.67), (0.78, 0.67))]:
        arrow(ax, start, end)

    action_x = [0.255, 0.305, 0.355, 0.405]
    action_labels = ["No-op", "Repair", "Delete", "Replace"]
    action_colors = ["#D7DEE8", "#CDE6C7", "#F6C9C9", "#FAD8A6"]
    for x, label, color in zip(action_x, action_labels, action_colors):
        rounded_box(ax, x, 0.31, 0.043, 0.105, label, fc=color, lw=0.9, fontsize=6.4)
        arrow(ax, (x + 0.022, 0.52), (x + 0.022, 0.415), color=COLORS["gray"])

    rounded_box(ax, 0.52, 0.31, 0.063, 0.105, "Cleaner", fc="#DFF0D8", lw=0.9, fontsize=6.4)
    rounded_box(ax, 0.595, 0.31, 0.063, 0.105, "Filter", fc="#F7D1D1", lw=0.9, fontsize=6.4)
    rounded_box(ax, 0.67, 0.31, 0.063, 0.105, "VE", fc="#FBE2B9", lw=0.9, fontsize=6.4)
    ax.text(0.62, 0.245, "operator families can be extended", ha="center", fontsize=7.2, color=COLORS["gray"])

    rounded_box(ax, 0.40, 0.06, 0.22, 0.13, "Utility Verifier\naccept / rollback / update", fc="#F3EAF7", fontsize=8.0)
    arrow(ax, (0.87, 0.52), (0.62, 0.19), color=COLORS["purple"], rad=-0.25)
    arrow(ax, (0.40, 0.125), (0.35, 0.52), color=COLORS["purple"], rad=-0.15)
    ax.text(0.50, 0.94, "DDPAgent separates what action to take from how operators execute it", ha="center", fontsize=10.5, weight="bold")

    fig.tight_layout(pad=0.2)
    fig.savefig(FIG_DIR / "ddpagent_framework_placeholder.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "ddpagent_framework_placeholder.png", bbox_inches="tight")
    plt.close(fig)


def make_instance_figure():
    fig, ax = plt.subplots(figsize=(3.50, 2.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    action_colors = {
        "Repair": "#55A868",
        "Delete": "#C44E52",
        "No-op": "#4C72B0",
        "Replace": "#DD8452",
    }
    rows = [
        ("style typo", "Repair", [r"$o_{rule}$", r"$o_{ctx}$"], "$\\ell_1=(p_1,A_{\\mathrm{style}},$\n$\\mathrm{Lager},o_{ctx})$"),
        ("harmful row", "Delete", [r"$o_{risk}$", r"$o_{filter}$"], "$\\ell_2=(\\mathrm{del},t_{42},$\n$o_{filter})$"),
        ("rare sample", "No-op", [r"$o_{valid}$", r"$o_{keep}$"], "$\\ell_3=(\\mathrm{keep},t_{17},$\n$o_{keep})$"),
        ("missing value", "Replace", [r"$o_{lookup}$", r"$o_{ve}$"], "$\\ell_4=(p_2,A_{\\mathrm{ibu}},$\n$35,o_{ve})$"),
    ]

    def node(x, y, w, h, text, fc, ec, color="#252A31", weight="normal", fs=6.35, linespacing=1.12):
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.010,rounding_size=0.018",
            linewidth=0.95,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            color=color,
            weight=weight,
            linespacing=linespacing,
        )
        return box

    ax.text(0.158, 0.935, "Dirty\nunits", ha="center", va="center", fontsize=7.05, weight="bold", color="#252A31", linespacing=0.9)
    ax.text(0.407, 0.935, "Action\nfamily", ha="center", va="center", fontsize=7.05, weight="bold", color="#252A31", linespacing=0.9)
    ax.text(0.636, 0.935, "Ordered\noperators", ha="center", va="center", fontsize=7.05, weight="bold", color="#252A31", linespacing=0.9)
    ax.text(0.885, 0.935, "Data\noperation", ha="center", va="center", fontsize=7.05, weight="bold", color="#252A31", linespacing=0.9)

    data_panel = FancyBboxPatch(
        (0.035, 0.100),
        0.250,
        0.785,
        boxstyle="round,pad=0.016,rounding_size=0.026",
        linewidth=1.05,
        edgecolor="#B7BDC7",
        facecolor="#FFFFFF",
    )
    ax.add_patch(data_panel)

    y_centers = [0.785, 0.585, 0.385, 0.185]
    for idx, ((issue, action_label, operators, operation), yc) in enumerate(zip(rows, y_centers), start=1):
        color = action_colors[action_label]
        ax.plot([0.062], [yc], marker="o", markersize=4.2, color=color, clip_on=False)
        ax.text(0.082, yc, f"$e_{idx}$  {issue}", ha="left", va="center", fontsize=6.45, color="#252A31")

        node(0.345, yc - 0.045, 0.130, 0.090, action_label, "#FFFFFF", color, color=color, weight="bold", fs=6.65)
        arrow(ax, (0.285, yc), (0.345, yc), color="#69707A")

        op_x = 0.555
        last_x = None
        for op_idx, op_text in enumerate(operators):
            x = op_x + op_idx * 0.095
            node(x, yc - 0.038, 0.064, 0.076, op_text, "#F7F8FA", "#B7BDC7", fs=7.05)
            if op_idx == 0:
                arrow(ax, (0.475, yc), (x - 0.010, yc), color="#69707A")
            else:
                arrow(ax, (last_x + 0.070, yc), (x - 0.010, yc), color="#69707A")
            last_x = x

        node(0.790, yc - 0.048, 0.195, 0.096, operation, "#FFFFFF", "#B7BDC7", fs=5.85, linespacing=0.92)
        arrow(ax, (last_x + 0.070, yc), (0.790, yc), color="#69707A")

    fig.tight_layout(pad=0.08)
    fig.savefig(FIG_DIR / "ddpagent_instance_flow.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "ddpagent_instance_flow.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "ddpagent_instance_flow.svg", bbox_inches="tight")
    plt.close(fig)


def make_evidence_figure():
    ads = pd.read_csv(OUT_FINAL / "adsclean" / "adsclean_summary.csv")
    base = pd.read_csv(OUT_FINAL / "baseline_eval" / "artificial" / "baseline_ml_summary.csv")
    hospital_ads = OUT_HOSPITAL_TARGET / "adsclean" / "adsclean_summary.csv"
    hospital_base = OUT_HOSPITAL_TARGET / "baseline_eval" / "artificial" / "baseline_ml_summary.csv"
    if hospital_ads.exists() and hospital_base.exists():
        ads_override = pd.read_csv(hospital_ads)
        base_override = pd.read_csv(hospital_base)
        ads = pd.concat([ads[ads["dataset"].ne("hospitals")], ads_override], ignore_index=True)
        base = pd.concat([base[base["dataset"].ne("hospitals")], base_override], ignore_index=True)
    artificial = ads[ads["scenario"].eq("artificial")]
    fixed_strategies = ["baran", "bigdansing", "holistic", "holoclean", "horizon", "uniclean_full"]

    dataset_order = ["beers", "flights", "hospitals", "rayyan", "tax"]
    labels = ["Beers", "Flights", "Hosp.", "Rayyan", "Tax-10K"]
    win_counts = []
    improve_counts = []
    rollback_counts = []
    action_mix = []
    for dataset in dataset_order:
        cur = artificial[artificial["dataset"].eq(dataset)]
        wins = improves = rollbacks = 0
        action_totals = {
            "no_action": 0,
            "repair_value": 0,
            "delete": 0,
            "replace_nearby": 0,
        }
        for _, row_data in cur.iterrows():
            rate = row_data["error_rate"]
            baselines = base[
                base["dataset"].eq(dataset)
                & base["error_rate"].eq(rate)
                & base["strategy"].isin(fixed_strategies)
            ]["downstream_fixed_after"].dropna()
            no_op = base[
                base["dataset"].eq(dataset)
                & base["error_rate"].eq(rate)
                & base["strategy"].eq("no_op")
            ]["downstream_fixed_after"].iloc[0]
            ddp = row_data["downstream_fixed_after"]
            wins += int(len(baselines) > 0 and ddp >= baselines.max() - 1e-12)
            improves += int(ddp > no_op + 1e-12)
            rollbacks += int(str(row_data["verifier_selected"]) == "no_op_rollback")
            counts = ast.literal_eval(row_data["action_counts"])
            for key in action_totals:
                action_totals[key] += int(counts.get(key, 0))
        win_counts.append(wins)
        improve_counts.append(improves)
        rollback_counts.append(rollbacks)
        total_actions = max(sum(action_totals.values()), 1)
        action_mix.append(
            {
                "No-op": action_totals["no_action"] / total_actions * 100.0,
                "Repair": action_totals["repair_value"] / total_actions * 100.0,
                "Delete": action_totals["delete"] / total_actions * 100.0,
                "Replace": action_totals["replace_nearby"] / total_actions * 100.0,
            }
        )

    fig, (ax, ax2) = plt.subplots(
        2,
        1,
        figsize=(3.55, 4.25),
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )
    x = list(range(len(dataset_order)))
    width = 0.24
    ax.bar([i - width for i in x], win_counts, width=width, color=COLORS["ddp"], edgecolor=COLORS["line"], linewidth=0.55, label=r"$\geq$ best fixed")
    ax.bar(x, improve_counts, width=width, color=COLORS["blue"], edgecolor=COLORS["line"], linewidth=0.55, label="> no-op")
    ax.bar([i + width for i in x], rollback_counts, width=width, color=COLORS["orange"], edgecolor=COLORS["line"], linewidth=0.55, label="rollback")
    ax.set_ylabel("# settings", fontsize=8.9)
    ax.set_ylim(0, 8.8)
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7.7)
    ax.grid(axis="y", linestyle="-", linewidth=0.35, alpha=0.26)
    for xpos, vals in [([i - width for i in x], win_counts), (x, improve_counts), ([i + width for i in x], rollback_counts)]:
        for xi, val in zip(xpos, vals):
            if val:
                ax.text(xi, val + 0.12, str(val), ha="center", va="bottom", fontsize=7.2)
    ax.text(0.5, -0.47, "(a) Outcome counts across injected settings", transform=ax.transAxes, ha="center", va="top", fontsize=8.2)
    ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.05), fontsize=7.25, columnspacing=0.58, handletextpad=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(width=0.9, length=3.2, labelsize=7.8)

    action_order = ["No-op", "Repair", "Delete", "Replace"]
    action_colors = {
        "No-op": "#9ECAE1",
        "Repair": COLORS["green"],
        "Delete": COLORS["ddp"],
        "Replace": COLORS["orange"],
    }
    bottoms = [0.0] * len(dataset_order)
    for action in action_order:
        values = [mix[action] for mix in action_mix]
        ax2.bar(
            x,
            values,
            bottom=bottoms,
            width=0.62,
            color=action_colors[action],
            edgecolor=COLORS["line"],
            linewidth=0.42,
            label=action,
        )
        for i, (bottom, value) in enumerate(zip(bottoms, values)):
            if value >= 13:
                ax2.text(i, bottom + value / 2, f"{value:.0f}", ha="center", va="center", fontsize=6.9, color="white")
        bottoms = [b + v for b, v in zip(bottoms, values)]
    ax2.set_ylabel("% actions", fontsize=8.9)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=7.7)
    ax2.grid(axis="y", linestyle="-", linewidth=0.35, alpha=0.26)
    ax2.text(0.5, -0.47, "(b) Average action allocation", transform=ax2.transAxes, ha="center", va="top", fontsize=8.2)
    ax2.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.05), fontsize=7.15, columnspacing=0.36, handletextpad=0.18)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax2.spines[side].set_linewidth(0.9)
    ax2.tick_params(width=0.9, length=3.2, labelsize=7.8)

    fig.subplots_adjust(left=0.16, right=0.985, top=0.935, bottom=0.13, hspace=0.84)
    fig.savefig(FIG_DIR / "ddpagent_selected_evidence.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "ddpagent_selected_evidence.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "ddpagent_selected_evidence.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    setup()
    make_instance_figure()
    make_evidence_figure()


if __name__ == "__main__":
    main()
