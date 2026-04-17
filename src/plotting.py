"""
plotting.py – Generate publication-ready plots for the quality–latency study.

All plots use matplotlib only (no seaborn).  Defaults are tuned for
readability on screen and in print.
"""

import os
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers / CI
import matplotlib.pyplot as plt
import numpy as np

from src.utils import log, model_name_to_id


# ================================================================== #
#  Style defaults                                                      #
# ================================================================== #

# One colour per model (colour-blind friendly palette)
MODEL_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # pink
    "#F0E442",  # yellow
    "#56B4E9",  # light blue
]

MODEL_MARKERS = ["o", "s", "D", "^", "v", "P"]

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})


# ================================================================== #
#  Public API                                                          #
# ================================================================== #

def generate_all_plots(
    summaries: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    """Entry point: create all plots and save to output_dir/plots/."""
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Split summaries by task
    summ_summaries = [s for s in summaries if s["task"] == "summarization"]
    qa_summaries = [s for s in summaries if s["task"] == "qa"]

    if summ_summaries:
        _plot_quality_vs_latency(
            summ_summaries,
            quality_key="rougeL",
            quality_label="ROUGE-L",
            title="Summarization: ROUGE-L vs. Latency",
            save_path=os.path.join(plot_dir, "summarization_quality_vs_latency.png"),
        )
        log("Saved summarization quality-vs-latency plot.")

    if qa_summaries:
        _plot_quality_vs_latency(
            qa_summaries,
            quality_key="f1",
            quality_label="Token F1",
            title="QA: Token F1 vs. Latency",
            save_path=os.path.join(plot_dir, "qa_quality_vs_latency.png"),
        )
        log("Saved QA quality-vs-latency plot.")

    if summaries:
        _plot_comparison_bar(summaries, output_dir=plot_dir)
        log("Saved comparison bar chart.")


# ================================================================== #
#  Scatter: quality vs latency                                         #
# ================================================================== #

def _plot_quality_vs_latency(
    summaries: List[Dict[str, Any]],
    quality_key: str,
    quality_label: str,
    title: str,
    save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, s in enumerate(summaries):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        marker = MODEL_MARKERS[i % len(MODEL_MARKERS)]
        label = s.get("model_id", s.get("model", f"model_{i}"))

        x = s["mean_latency_s"]
        y = s[quality_key]

        # Compute error bar extents from bootstrap CIs if available
        x_lo = s.get("mean_latency_ci_lower")
        x_hi = s.get("mean_latency_ci_upper")
        y_lo = s.get(f"{quality_key}_ci_lower")
        y_hi = s.get(f"{quality_key}_ci_upper")

        xerr = None
        yerr = None
        if x_lo is not None and x_hi is not None:
            xerr = [[x - x_lo], [x_hi - x]]
        if y_lo is not None and y_hi is not None:
            yerr = [[y - y_lo], [y_hi - y]]

        ax.errorbar(
            x, y,
            xerr=xerr, yerr=yerr,
            fmt=marker,
            color=color,
            markersize=10,
            label=label,
            capsize=4,
            capthick=1.5,
            elinewidth=1.5,
            markeredgecolor="black",
            markeredgewidth=0.5,
            zorder=3,
        )

    ax.set_xlabel("Mean Latency (seconds)")
    ax.set_ylabel(quality_label)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.savefig(save_path)
    plt.close(fig)


# ================================================================== #
#  Bar chart: side-by-side model comparison                            #
# ================================================================== #

def _plot_comparison_bar(
    summaries: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    """
    Create a grouped bar chart showing key metrics across models,
    with separate subplots for each task.
    """
    tasks = sorted(set(s["task"] for s in summaries))

    for task in tasks:
        task_summaries = [s for s in summaries if s["task"] == task]
        if not task_summaries:
            continue

        if task == "summarization":
            metric_keys = ["rouge1", "rouge2", "rougeL"]
            metric_labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
        else:
            metric_keys = ["exact_match", "f1"]
            metric_labels = ["Exact Match", "Token F1"]

        model_labels = [s.get("model_id", "?") for s in task_summaries]
        n_models = len(model_labels)
        n_metrics = len(metric_keys)

        x = np.arange(n_models)
        width = 0.8 / n_metrics

        fig, ax = plt.subplots(figsize=(max(8, n_models * 2.5), 5))

        for j, (mk, ml) in enumerate(zip(metric_keys, metric_labels)):
            vals = [s.get(mk, 0) for s in task_summaries]
            offset = (j - n_metrics / 2 + 0.5) * width
            bars = ax.bar(
                x + offset, vals, width,
                label=ml,
                color=MODEL_COLORS[j % len(MODEL_COLORS)],
                edgecolor="black",
                linewidth=0.5,
            )
            # Value labels on bars
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=15, ha="right")
        ax.set_ylabel("Score")
        ax.set_title(f"{task.capitalize()} – Quality Metrics by Model")
        ax.legend(loc="best")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

        save_path = os.path.join(output_dir, f"{task}_comparison.png")
        fig.savefig(save_path)
        plt.close(fig)
