#!/usr/bin/env python3
"""
poster.py – Generate a conference-style poster for the final project presentation.

Usage:
    python3 poster.py

Output: outputs/demo/poster.png
"""

import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


# ================================================================== #
#  Config                                                              #
# ================================================================== #

OUTPUT_PATH = os.path.join("outputs", "demo", "poster.png")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

MODEL_COLORS = {
    "TinyLlama-1.1B": "#0072B2",
    "Phi-3-mini-3.8B": "#D55E00",
    "Mistral-7B": "#009E73",
}
MODELS = ["TinyLlama-1.1B", "Phi-3-mini-3.8B", "Mistral-7B"]
PARAMS = {"TinyLlama-1.1B": "1.1B", "Phi-3-mini-3.8B": "3.8B", "Mistral-7B": "7.2B"}

# Results data
SUMM = {
    "TinyLlama-1.1B":   {"rouge1": 0.231, "rouge2": 0.067, "rougeL": 0.174, "lat": 1.82, "tps": 48.3, "mem": 2.4},
    "Phi-3-mini-3.8B":  {"rouge1": 0.324, "rouge2": 0.119, "rougeL": 0.249, "lat": 4.57, "tps": 31.2, "mem": 7.7},
    "Mistral-7B":       {"rouge1": 0.359, "rouge2": 0.140, "rougeL": 0.279, "lat": 8.34, "tps": 18.7, "mem": 13.9},
}
QA = {
    "TinyLlama-1.1B":   {"em": 0.124, "f1": 0.257, "lat": 0.94, "tps": 52.1},
    "Phi-3-mini-3.8B":  {"em": 0.412, "f1": 0.583, "lat": 2.31, "tps": 34.8},
    "Mistral-7B":       {"em": 0.489, "f1": 0.651, "lat": 4.12, "tps": 21.4},
}


def main():
    # Poster dimensions (48x36 inches at 100 dpi = 4800x3600 px)
    fig = plt.figure(figsize=(48, 36), facecolor="white")

    # Main grid: header row + 3 content rows
    outer = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[0.12, 0.32, 0.32, 0.24],
                              hspace=0.08, top=0.97, bottom=0.02, left=0.03, right=0.97)

    # ============================================================== #
    #  HEADER                                                         #
    # ============================================================== #
    header_ax = fig.add_subplot(outer[0])
    header_ax.axis("off")

    # Background banner
    banner = FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.01",
                             facecolor="#1a1a2e", edgecolor="none",
                             transform=header_ax.transAxes)
    header_ax.add_patch(banner)

    header_ax.text(0.5, 0.65, "When Is Smaller Better?",
                   transform=header_ax.transAxes, fontsize=52, fontweight="bold",
                   color="white", ha="center", va="center")
    header_ax.text(0.5, 0.28, "Evaluating Quality-Latency Tradeoffs in Instruction-Tuned Language Models",
                   transform=header_ax.transAxes, fontsize=30, color="#cccccc",
                   ha="center", va="center")
    header_ax.text(0.5, 0.05, "Jason Gao  |  CSCI 5922: Neural Networks and Deep Learning  |  Spring 2026",
                   transform=header_ax.transAxes, fontsize=22, color="#999999",
                   ha="center", va="center")

    # ============================================================== #
    #  ROW 1: Motivation | Models & Setup | Approach                  #
    # ============================================================== #
    row1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.06)

    # --- Motivation ---
    ax = fig.add_subplot(row1[0])
    ax.axis("off")
    _draw_section_box(ax, "#e8f0fe", "#1a73e8")
    ax.text(0.5, 0.95, "Motivation", transform=ax.transAxes, fontsize=28,
            fontweight="bold", ha="center", va="top", color="#1a1a2e")
    motivation_text = (
        "Large Language Models (LLMs) achieve impressive results "
        "on NLP tasks, but they come with significant costs:\n\n"
        "     Slower inference speed\n"
        "     Higher GPU memory requirements\n"
        "     Greater energy consumption\n\n"
        "Practitioners face a critical question:\n\n"
        "When can a smaller, faster model deliver\n"
        "\"good enough\" results compared to a\n"
        "larger one?\n\n"
        "Existing benchmarks (HELM, Open LLM Leaderboard)\n"
        "rank models by quality alone. They don't show\n"
        "the quality-per-second tradeoff that matters\n"
        "for real-world deployment decisions."
    )
    ax.text(0.08, 0.85, motivation_text, transform=ax.transAxes, fontsize=18,
            va="top", ha="left", linespacing=1.6, fontfamily="sans-serif")

    # --- Models & Setup ---
    ax = fig.add_subplot(row1[1])
    ax.axis("off")
    _draw_section_box(ax, "#fef7e0", "#f9ab00")
    ax.text(0.5, 0.95, "Experimental Setup", transform=ax.transAxes, fontsize=28,
            fontweight="bold", ha="center", va="top", color="#1a1a2e")

    # Model table
    models_header = ["Model", "Parameters", "Type"]
    models_data = [
        ["TinyLlama-1.1B", "1.1 Billion", "Chat-tuned"],
        ["Phi-3-mini", "3.8 Billion", "Instruction-tuned"],
        ["Mistral-7B", "7.2 Billion", "Instruction-tuned"],
    ]
    table = ax.table(
        cellText=models_data, colLabels=models_header,
        cellLoc="center", loc="upper center",
        bbox=[0.05, 0.62, 0.9, 0.24],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1, 2.0)
    for j in range(3):
        table[0, j].set_facecolor("#1a1a2e")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i, m in enumerate(MODELS):
        table[i+1, 0].set_facecolor(MODEL_COLORS[m] + "30")

    setup_text = (
        "Tasks:\n"
        "  Summarization — XSum dataset (ROUGE-1/2/L)\n"
        "  Question Answering — SQuAD v1.1 (EM / F1)\n\n"
        "Controls:\n"
        "  300 examples per task, fixed seed (42)\n"
        "  Greedy decoding (temperature = 0)\n"
        "  Per-example latency with time.perf_counter()\n"
        "  3-step warmup before each model\n"
        "  Models loaded one at a time to control memory"
    )
    ax.text(0.08, 0.55, setup_text, transform=ax.transAxes, fontsize=17,
            va="top", ha="left", linespacing=1.6, fontfamily="sans-serif")

    # --- Approach ---
    ax = fig.add_subplot(row1[2])
    ax.axis("off")
    _draw_section_box(ax, "#e8f5e9", "#34a853")
    ax.text(0.5, 0.95, "Approach", transform=ax.transAxes, fontsize=28,
            fontweight="bold", ha="center", va="top", color="#1a1a2e")

    approach_text = (
        "Pipeline Architecture:\n\n"
        "  1. Load dataset (XSum or SQuAD)\n"
        "  2. Load model + tokenizer\n"
        "  3. Warm up with dummy generations\n"
        "  4. For each example:\n"
        "       Build chat-formatted prompt\n"
        "       Time the generation call\n"
        "       Score output (ROUGE or EM/F1)\n"
        "  5. Compute aggregate metrics + CIs\n"
        "  6. Unload model, repeat for next\n"
        "  7. Pareto frontier analysis\n\n"
        "Key Design Decisions:\n\n"
        "  Batch size = 1 for consistent timing\n"
        "  Chat templates for fair comparison\n"
        "  Bootstrap CIs (n=1000) for significance\n"
        "  Answer extraction heuristics for QA"
    )
    ax.text(0.08, 0.85, approach_text, transform=ax.transAxes, fontsize=17,
            va="top", ha="left", linespacing=1.5, fontfamily="sans-serif")

    # ============================================================== #
    #  ROW 2: Results plots                                           #
    # ============================================================== #
    row2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.08)

    # --- Quality vs Latency scatter ---
    ax = fig.add_subplot(row2[0])
    _draw_section_box(ax, "#ffffff", "#cccccc")
    # Summarization subplot
    for m in MODELS:
        ax.scatter(SUMM[m]["lat"], SUMM[m]["rougeL"], s=400, color=MODEL_COLORS[m],
                   marker="o", edgecolors="black", linewidth=1.5, zorder=5,
                   label=f"{m} ({PARAMS[m]})")
    pts = sorted([(SUMM[m]["lat"], SUMM[m]["rougeL"]) for m in MODELS])
    ax.plot([p[0] for p in pts], [p[1] for p in pts], '--', color='gray', alpha=0.5, lw=2)
    ax.set_xlabel("Mean Latency (seconds)", fontsize=16)
    ax.set_ylabel("ROUGE-L", fontsize=16)
    ax.set_title("Summarization: Quality vs. Latency", fontsize=22, fontweight="bold", pad=15)
    ax.legend(fontsize=14, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=14)
    ax.annotate("4.6x faster\n5 pts lower", xy=(1.82, 0.174), xytext=(4.5, 0.155),
                fontsize=14, ha="center",
                arrowprops=dict(arrowstyle="->", color="gray", lw=2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

    # --- Quality bars ---
    ax = fig.add_subplot(row2[1])
    _draw_section_box(ax, "#ffffff", "#cccccc")
    x = np.arange(len(MODELS))
    w = 0.25
    colors_metric = ["#0072B2", "#D55E00", "#009E73"]
    for j, (mk, ml) in enumerate(zip(["rouge1", "rouge2", "rougeL"], ["ROUGE-1", "ROUGE-2", "ROUGE-L"])):
        vals = [SUMM[m][mk] for m in MODELS]
        bars = ax.bar(x + (j - 1) * w, vals, w, label=ml, color=colors_metric[j],
                      edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=15)
    ax.set_ylabel("Score", fontsize=16)
    ax.set_title("Summarization Quality", fontsize=22, fontweight="bold", pad=15)
    ax.legend(fontsize=14)
    ax.set_ylim(0, 0.45)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(labelsize=14)

    # --- QA quality + latency ---
    ax = fig.add_subplot(row2[2])
    _draw_section_box(ax, "#ffffff", "#cccccc")
    for m in MODELS:
        ax.scatter(QA[m]["lat"], QA[m]["f1"], s=400, color=MODEL_COLORS[m],
                   marker="s", edgecolors="black", linewidth=1.5, zorder=5,
                   label=f"{m} ({PARAMS[m]})")
    pts = sorted([(QA[m]["lat"], QA[m]["f1"]) for m in MODELS])
    ax.plot([p[0] for p in pts], [p[1] for p in pts], '--', color='gray', alpha=0.5, lw=2)
    ax.set_xlabel("Mean Latency (seconds)", fontsize=16)
    ax.set_ylabel("Token F1", fontsize=16)
    ax.set_title("QA: Quality vs. Latency", fontsize=22, fontweight="bold", pad=15)
    ax.legend(fontsize=14, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=14)
    ax.annotate("4.4x faster\n39.5 pts lower", xy=(0.94, 0.257), xytext=(2.5, 0.32),
                fontsize=14, ha="center",
                arrowprops=dict(arrowstyle="->", color="gray", lw=2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

    # ============================================================== #
    #  ROW 3: Efficiency | Key Findings | Conclusions                 #
    # ============================================================== #
    row3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[3], wspace=0.06)

    # --- Efficiency ---
    ax = fig.add_subplot(row3[0])
    _draw_section_box(ax, "#ffffff", "#cccccc")
    x = np.arange(len(MODELS))
    mems = [SUMM[m]["mem"] for m in MODELS]
    bars = ax.bar(x, mems, 0.5, color=[MODEL_COLORS[m] for m in MODELS],
                  edgecolor="black", linewidth=1)
    for bar, v in zip(bars, mems):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{v:.1f} GB", ha="center", va="bottom", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=14)
    ax.set_ylabel("Peak VRAM (GB)", fontsize=16)
    ax.set_title("GPU Memory Usage", fontsize=22, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(labelsize=14)

    # --- Key Findings ---
    ax = fig.add_subplot(row3[1])
    ax.axis("off")
    _draw_section_box(ax, "#fff3e0", "#e65100")
    ax.text(0.5, 0.95, "Key Findings", transform=ax.transAxes, fontsize=28,
            fontweight="bold", ha="center", va="top", color="#1a1a2e")

    findings = [
        ("1", "Phi-3 is the sweet spot",
         "Achieves 89% of Mistral-7B's quality\nat only 45% of the latency cost."),
        ("2", "Task complexity matters",
         "Small models lose 10 pts on summarization\nbut 39 pts on QA — harder tasks\nexpose bigger gaps."),
        ("3", "Small models can't follow instructions",
         "TinyLlama often ignores the question\nand generates irrelevant text instead\nof extracting the answer."),
    ]
    y_start = 0.82
    for num, title, desc in findings:
        ax.text(0.08, y_start, f"{num}.", transform=ax.transAxes, fontsize=26,
                fontweight="bold", color="#e65100", va="top")
        ax.text(0.15, y_start, title, transform=ax.transAxes, fontsize=20,
                fontweight="bold", va="top", color="#1a1a2e")
        ax.text(0.15, y_start - 0.08, desc, transform=ax.transAxes, fontsize=16,
                va="top", color="#333333", linespacing=1.4)
        y_start -= 0.32

    # --- Conclusions ---
    ax = fig.add_subplot(row3[2])
    ax.axis("off")
    _draw_section_box(ax, "#f3e5f5", "#7b1fa2")
    ax.text(0.5, 0.95, "Conclusions & Future Work", transform=ax.transAxes, fontsize=28,
            fontweight="bold", ha="center", va="top", color="#1a1a2e")

    conclusions_text = (
        "Conclusions:\n\n"
        "  For latency-sensitive applications,\n"
        "  Phi-3-mini offers the best balance of\n"
        "  quality and speed.\n\n"
        "  The \"right\" model depends on the task:\n"
        "  simpler tasks tolerate smaller models,\n"
        "  but complex reasoning tasks require\n"
        "  larger ones.\n\n"
        "Future Directions:\n\n"
        "  Extend to quantized models (INT8/INT4)\n"
        "  Test on more diverse tasks\n"
        "  Measure energy consumption\n"
        "  Evaluate on edge devices"
    )
    ax.text(0.08, 0.85, conclusions_text, transform=ax.transAxes, fontsize=17,
            va="top", ha="left", linespacing=1.5, fontfamily="sans-serif")

    # Save
    fig.savefig(OUTPUT_PATH, dpi=100, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Poster saved to: {OUTPUT_PATH}")


def _draw_section_box(ax, facecolor, edgecolor):
    """Draw a rounded box around a section."""
    rect = FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                           boxstyle="round,pad=0.02",
                           facecolor=facecolor, edgecolor=edgecolor,
                           linewidth=2.5, transform=ax.transAxes,
                           zorder=0)
    ax.add_patch(rect)


if __name__ == "__main__":
    main()
