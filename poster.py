#!/usr/bin/env python3
"""
poster.py – Generate a clean, audience-friendly poster for presentation.

Usage:
    python3 poster.py

Output: outputs/demo/poster.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUTPUT_PATH = os.path.join("outputs", "demo", "poster.png")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Colors
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
DARK = "#1a1a2e"
RED = "#c62828"

MODEL_COLORS = [BLUE, ORANGE, GREEN]
MODELS_SHORT = ["TinyLlama\n1.1B", "Phi-3\n3.8B", "Mistral\n7B"]

# Data
SUMM_ROUGEL = [0.174, 0.249, 0.279]
SUMM_LAT = [1.82, 4.57, 8.34]
QA_F1 = [0.257, 0.583, 0.651]
QA_LAT = [0.94, 2.31, 4.12]
MEMORY_GB = [2.4, 7.7, 13.9]
THROUGHPUT = [48.3, 31.2, 18.7]


def main():
    fig = plt.figure(figsize=(60, 40), facecolor="white")

    outer = gridspec.GridSpec(
        4, 1, figure=fig,
        height_ratios=[0.08, 0.32, 0.32, 0.28],
        hspace=0.05, top=0.98, bottom=0.02, left=0.025, right=0.975
    )

    # ============================================================ #
    #  HEADER                                                       #
    # ============================================================ #
    ax = fig.add_subplot(outer[0])
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.008",
        facecolor=DARK, edgecolor="none", transform=ax.transAxes
    ))
    ax.text(0.5, 0.72, "When Is Smaller Better?",
            transform=ax.transAxes, fontsize=72, fontweight="bold",
            color="white", ha="center", va="center")
    ax.text(0.5, 0.32, "We tested whether cheap, fast AI models can replace expensive ones — and found the surprising sweet spot.",
            transform=ax.transAxes, fontsize=32, color="#bbbbbb",
            ha="center", va="center", style="italic")
    ax.text(0.5, 0.06, "Alex Zhou  &  Jason Gao   |   CSCI 5922: Neural Networks & Deep Learning   |   Spring 2026",
            transform=ax.transAxes, fontsize=26, color="#888888", ha="center", va="center")

    # ============================================================ #
    #  ROW 1: Problem | What We Did | Models                        #
    # ============================================================ #
    row1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.04)

    # --- THE PROBLEM ---
    ax = fig.add_subplot(row1[0])
    ax.axis("off")
    _box(ax, "#eef4ff", BLUE)
    ax.text(0.5, 0.95, "The Problem", transform=ax.transAxes,
            fontsize=42, fontweight="bold", ha="center", va="top", color=DARK)

    ax.add_patch(FancyBboxPatch(
        (0.06, 0.74), 0.88, 0.14, boxstyle="round,pad=0.02",
        facecolor="white", edgecolor=BLUE, linewidth=3, transform=ax.transAxes
    ))
    ax.text(0.5, 0.81, "Bigger AI = better answers, but slower & costlier.\nIs the upgrade always worth it?",
            transform=ax.transAxes, fontsize=26, ha="center", va="center",
            color=DARK, linespacing=1.4, fontweight="bold")

    lines = [
        (0.76, "Large model", "Great quality, but slow & expensive", GREEN, RED),
        (0.58, "Small model", "Fast & cheap, but are answers good enough?", BLUE, ORANGE),
    ]
    for y_pos, label, desc, col1, col2 in lines:
        ax.add_patch(FancyBboxPatch(
            (0.06, y_pos - 0.06), 0.88, 0.14, boxstyle="round,pad=0.02",
            facecolor=col1 + "15", edgecolor=col1, linewidth=2, transform=ax.transAxes
        ))
        ax.text(0.12, y_pos + 0.04, label, transform=ax.transAxes,
                fontsize=28, fontweight="bold", va="center", color=col1)
        ax.text(0.12, y_pos - 0.025, desc, transform=ax.transAxes,
                fontsize=22, va="center", color="#333333")

    ax.text(0.5, 0.36, "Existing leaderboards only rank quality.\nThey ignore speed and cost —", transform=ax.transAxes,
            fontsize=24, ha="center", va="center", color="#444444", linespacing=1.5)
    ax.text(0.5, 0.22, "the factors that actually matter\nfor real-world deployment.", transform=ax.transAxes,
            fontsize=26, ha="center", va="center", color=DARK, fontweight="bold", linespacing=1.4)

    # --- WHAT WE DID ---
    ax = fig.add_subplot(row1[1])
    ax.axis("off")
    _box(ax, "#fff8e1", ORANGE)
    ax.text(0.5, 0.95, "What We Did", transform=ax.transAxes,
            fontsize=42, fontweight="bold", ha="center", va="top", color=DARK)

    steps = [
        ("1", "Picked 3 AI models of different sizes",
         "Small (1.1B) · Medium (3.8B) · Large (7B)"),
        ("2", "Gave them 600 real tasks",
         "300 summaries + 300 reading questions"),
        ("3", "Measured quality AND speed",
         "How good? How fast? How much memory?"),
        ("4", "Found the best bang for your buck",
         "Which model gives the best quality-per-second?"),
    ]
    y = 0.84
    for num, title, desc in steps:
        ax.add_patch(FancyBboxPatch(
            (0.06, y - 0.01), 0.07, 0.065, boxstyle="round,pad=0.008",
            facecolor=ORANGE, edgecolor="none", transform=ax.transAxes
        ))
        ax.text(0.095, y + 0.02, num, transform=ax.transAxes, fontsize=30,
                fontweight="bold", ha="center", va="center", color="white")
        ax.text(0.17, y + 0.035, title, transform=ax.transAxes, fontsize=26,
                fontweight="bold", va="center", color=DARK)
        ax.text(0.17, y - 0.02, desc, transform=ax.transAxes, fontsize=22,
                va="center", color="#555555")
        y -= 0.19

    # --- THE MODELS ---
    ax = fig.add_subplot(row1[2])
    ax.axis("off")
    _box(ax, "#e8f5e9", GREEN)
    ax.text(0.5, 0.95, "The Three Models", transform=ax.transAxes,
            fontsize=42, fontweight="bold", ha="center", va="top", color=DARK)

    models_info = [
        ("TinyLlama", "1.1 Billion parameters", BLUE, "The lightweight.", "2.4 GB"),
        ("Phi-3 Mini", "3.8 Billion parameters", ORANGE, "The middle ground.", "7.7 GB"),
        ("Mistral-7B", "7.2 Billion parameters", GREEN, "The heavyweight.", "13.9 GB"),
    ]
    y = 0.83
    for name, params, color, desc, mem in models_info:
        ax.add_patch(FancyBboxPatch(
            (0.05, y - 0.07), 0.9, 0.19, boxstyle="round,pad=0.02",
            facecolor="white", edgecolor=color, linewidth=4, transform=ax.transAxes
        ))
        ax.text(0.12, y + 0.07, name, transform=ax.transAxes, fontsize=32,
                fontweight="bold", va="center", color=color)
        ax.text(0.92, y + 0.07, mem, transform=ax.transAxes, fontsize=22,
                va="center", ha="right", color="#666666", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color + "20", edgecolor="none"))
        ax.text(0.12, y + 0.01, params, transform=ax.transAxes, fontsize=22,
                va="center", color="#666666")
        ax.text(0.12, y - 0.04, desc, transform=ax.transAxes, fontsize=24,
                va="center", color="#333333", fontweight="bold")
        y -= 0.28

    # ============================================================ #
    #  ROW 2: Results plots                                         #
    # ============================================================ #
    row2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.06)

    # --- Summarization scatter ---
    ax = fig.add_subplot(row2[0])
    for i in range(3):
        ax.scatter(SUMM_LAT[i], SUMM_ROUGEL[i] * 100, s=900,
                   color=MODEL_COLORS[i], marker="o",
                   edgecolors="black", linewidth=2.5, zorder=5,
                   label=["TinyLlama 1.1B", "Phi-3 3.8B", "Mistral 7B"][i])
    ax.plot(SUMM_LAT, [r * 100 for r in SUMM_ROUGEL], '--', color='gray', alpha=0.4, lw=2.5)
    ax.set_xlabel("Time per answer (seconds)", fontsize=24, labelpad=12)
    ax.set_ylabel("Summary Quality (%)", fontsize=24, labelpad=12)
    ax.set_title("Summarization", fontsize=34, fontweight="bold", pad=20)
    ax.legend(fontsize=20, loc="lower right", markerscale=0.7)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=20)
    ax.annotate("4.6x faster\nonly 10 pts lower",
                xy=(SUMM_LAT[0], SUMM_ROUGEL[0] * 100),
                xytext=(5.0, 14), fontsize=20, ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=3),
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#e3f2fd", edgecolor=BLUE, lw=2))

    # --- QA scatter ---
    ax = fig.add_subplot(row2[1])
    for i in range(3):
        ax.scatter(QA_LAT[i], QA_F1[i] * 100, s=900,
                   color=MODEL_COLORS[i], marker="s",
                   edgecolors="black", linewidth=2.5, zorder=5,
                   label=["TinyLlama 1.1B", "Phi-3 3.8B", "Mistral 7B"][i])
    ax.plot(QA_LAT, [f * 100 for f in QA_F1], '--', color='gray', alpha=0.4, lw=2.5)
    ax.set_xlabel("Time per answer (seconds)", fontsize=24, labelpad=12)
    ax.set_ylabel("Answer Accuracy (%)", fontsize=24, labelpad=12)
    ax.set_title("Question Answering", fontsize=34, fontweight="bold", pad=20)
    ax.legend(fontsize=20, loc="lower right", markerscale=0.7)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=20)
    ax.annotate("TinyLlama can't\nfollow instructions!",
                xy=(QA_LAT[0], QA_F1[0] * 100),
                xytext=(2.5, 32), fontsize=20, ha="center", fontweight="bold", color=RED,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=3),
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffebee", edgecolor=RED, lw=2))

    # --- Speed & Memory ---
    ax = fig.add_subplot(row2[2])
    x = np.arange(3)
    w = 0.32
    bars1 = ax.bar(x - w/2, THROUGHPUT, w, color=MODEL_COLORS,
                   edgecolor="black", linewidth=1.5, label="Speed (tokens/sec)")
    for bar, v in zip(bars1, THROUGHPUT):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{v:.0f}", ha="center", va="bottom", fontsize=22, fontweight="bold")

    ax2 = ax.twinx()
    bars2 = ax2.bar(x + w/2, MEMORY_GB, w, color=MODEL_COLORS,
                    edgecolor="black", linewidth=1.5, alpha=0.35, hatch="///")
    for bar, v in zip(bars2, MEMORY_GB):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{v:.1f}GB", ha="center", va="bottom", fontsize=19, color="#444444", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["TinyLlama\n1.1B", "Phi-3\n3.8B", "Mistral\n7B"], fontsize=20)
    ax.set_ylabel("Speed (tokens/sec)", fontsize=22, labelpad=12)
    ax2.set_ylabel("Memory (GB)", fontsize=22, labelpad=12)
    ax.set_title("Speed vs. Memory Cost", fontsize=34, fontweight="bold", pad=20)
    ax.tick_params(labelsize=18)
    ax2.tick_params(labelsize=18)
    ax.grid(axis="y", alpha=0.2)

    # ============================================================ #
    #  ROW 3: Three Takeaways                                       #
    # ============================================================ #
    row3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[3], wspace=0.04)

    # --- TAKEAWAY 1 ---
    ax = fig.add_subplot(row3[0])
    ax.axis("off")
    _box(ax, "#e3f2fd", BLUE)
    ax.text(0.5, 0.94, "TAKEAWAY #1", transform=ax.transAxes,
            fontsize=28, fontweight="bold", ha="center", va="top", color=BLUE)
    ax.text(0.5, 0.72, "Phi-3 is the\nsweet spot", transform=ax.transAxes,
            fontsize=46, fontweight="bold", ha="center", va="center", color=DARK,
            linespacing=1.2)
    ax.add_patch(FancyBboxPatch(
        (0.08, 0.22), 0.84, 0.28, boxstyle="round,pad=0.03",
        facecolor="white", edgecolor=BLUE, linewidth=3, transform=ax.transAxes
    ))
    ax.text(0.5, 0.40, "89% of the quality", transform=ax.transAxes,
            fontsize=30, ha="center", va="center", color=DARK, fontweight="bold")
    ax.text(0.5, 0.30, "at 45% of the cost", transform=ax.transAxes,
            fontsize=30, ha="center", va="center", color=ORANGE, fontweight="bold")
    ax.text(0.5, 0.12, "Best quality-per-second of all models.",
            transform=ax.transAxes, fontsize=22, ha="center", va="center",
            color="#666666", style="italic")

    # --- TAKEAWAY 2 ---
    ax = fig.add_subplot(row3[1])
    ax.axis("off")
    _box(ax, "#fff8e1", ORANGE)
    ax.text(0.5, 0.94, "TAKEAWAY #2", transform=ax.transAxes,
            fontsize=28, fontweight="bold", ha="center", va="top", color=ORANGE)
    ax.text(0.5, 0.72, "The task\nmatters a lot", transform=ax.transAxes,
            fontsize=46, fontweight="bold", ha="center", va="center", color=DARK,
            linespacing=1.2)
    ax.add_patch(FancyBboxPatch(
        (0.08, 0.22), 0.84, 0.28, boxstyle="round,pad=0.03",
        facecolor="white", edgecolor=ORANGE, linewidth=3, transform=ax.transAxes
    ))
    ax.text(0.5, 0.42, "Summarization gap:  10 pts", transform=ax.transAxes,
            fontsize=28, ha="center", va="center", color=GREEN, fontweight="bold")
    ax.text(0.5, 0.30, "Question answering gap:  39 pts", transform=ax.transAxes,
            fontsize=28, ha="center", va="center", color=RED, fontweight="bold")
    ax.text(0.5, 0.12, "Harder tasks expose bigger gaps.",
            transform=ax.transAxes, fontsize=22, ha="center", va="center",
            color="#666666", style="italic")

    # --- TAKEAWAY 3 ---
    ax = fig.add_subplot(row3[2])
    ax.axis("off")
    _box(ax, "#e8f5e9", GREEN)
    ax.text(0.5, 0.94, "TAKEAWAY #3", transform=ax.transAxes,
            fontsize=28, fontweight="bold", ha="center", va="top", color=GREEN)
    ax.text(0.5, 0.72, "Small models\ndon't listen", transform=ax.transAxes,
            fontsize=46, fontweight="bold", ha="center", va="center", color=DARK,
            linespacing=1.2)
    ax.add_patch(FancyBboxPatch(
        (0.08, 0.18), 0.84, 0.34, boxstyle="round,pad=0.03",
        facecolor="white", edgecolor=GREEN, linewidth=3, transform=ax.transAxes
    ))
    ax.text(0.5, 0.46, 'Q: "When was the Eiffel Tower built?"', transform=ax.transAxes,
            fontsize=24, ha="center", va="center", color="#555555")
    ax.text(0.5, 0.37, 'Mistral:  "1889"', transform=ax.transAxes,
            fontsize=28, ha="center", va="center", color=GREEN, fontweight="bold")
    ax.text(0.5, 0.26, 'TinyLlama:  "The Eiffel Tower\nis a famous landmark..."', transform=ax.transAxes,
            fontsize=26, ha="center", va="center", color=RED, fontweight="bold", linespacing=1.3)
    ax.text(0.5, 0.10, "Smaller models ramble instead of answering.",
            transform=ax.transAxes, fontsize=22, ha="center", va="center",
            color="#666666", style="italic")

    fig.savefig(OUTPUT_PATH, dpi=100, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Poster saved to: {OUTPUT_PATH}")


def _box(ax, facecolor, edgecolor):
    ax.add_patch(FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0.02",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=3.5, transform=ax.transAxes, zorder=0
    ))


if __name__ == "__main__":
    main()
