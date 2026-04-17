#!/usr/bin/env python3
"""
poster.py – Generate a clean, audience-friendly poster for presentation.

Designed for a general audience (parents, friends, employers).
Plain language, big visuals, clear punchlines.

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
LIGHT_BG = "#f8f9fa"
ACCENT = "#e65100"

MODEL_COLORS = {"TinyLlama\n1.1B": BLUE, "Phi-3\n3.8B": ORANGE, "Mistral\n7B": GREEN}
MODELS = ["TinyLlama\n1.1B", "Phi-3\n3.8B", "Mistral\n7B"]

# Data
SUMM_ROUGEL = [0.174, 0.249, 0.279]
SUMM_LAT = [1.82, 4.57, 8.34]
QA_F1 = [0.257, 0.583, 0.651]
QA_LAT = [0.94, 2.31, 4.12]
MEMORY_GB = [2.4, 7.7, 13.9]
THROUGHPUT = [48.3, 31.2, 18.7]


def main():
    fig = plt.figure(figsize=(48, 36), facecolor="white")

    # Overall layout: header, then 3 rows
    outer = gridspec.GridSpec(
        5, 1, figure=fig,
        height_ratios=[0.10, 0.005, 0.30, 0.30, 0.25],
        hspace=0.04, top=0.98, bottom=0.02, left=0.03, right=0.97
    )

    # ============================================================ #
    #  HEADER                                                       #
    # ============================================================ #
    ax = fig.add_subplot(outer[0])
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.01",
        facecolor=DARK, edgecolor="none", transform=ax.transAxes
    ))
    ax.text(0.5, 0.7, "When Is Smaller Better?",
            transform=ax.transAxes, fontsize=58, fontweight="bold",
            color="white", ha="center", va="center", fontfamily="sans-serif")
    ax.text(0.5, 0.3, "We tested whether cheap, fast AI models can replace expensive ones — and found the surprising sweet spot.",
            transform=ax.transAxes, fontsize=26, color="#bbbbbb",
            ha="center", va="center", fontfamily="sans-serif", style="italic")
    ax.text(0.5, 0.05, "Alex Zhou  &  Jason Gao   |   CSCI 5922: Neural Networks & Deep Learning   |   Spring 2026",
            transform=ax.transAxes, fontsize=20, color="#888888",
            ha="center", va="center", fontfamily="sans-serif")

    # Divider
    ax = fig.add_subplot(outer[1])
    ax.axis("off")

    # ============================================================ #
    #  ROW 1: The Problem | What We Did | The Models                #
    # ============================================================ #
    row1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.05)

    # --- THE PROBLEM ---
    ax = fig.add_subplot(row1[0])
    ax.axis("off")
    _section_box(ax, "#eef4ff", BLUE)
    ax.text(0.5, 0.94, "The Problem", transform=ax.transAxes,
            fontsize=32, fontweight="bold", ha="center", va="top", color=DARK)
    ax.add_patch(FancyBboxPatch(
        (0.06, 0.70), 0.88, 0.18, boxstyle="round,pad=0.02",
        facecolor="white", edgecolor=BLUE, linewidth=2, transform=ax.transAxes
    ))
    ax.text(0.5, 0.79, "AI language models are getting bigger and better —\nbut bigger also means slower and more expensive.",
            transform=ax.transAxes, fontsize=20, ha="center", va="center",
            color=DARK, linespacing=1.5, fontweight="bold")

    problem_text = (
        "Companies deploying AI face a dilemma:\n\n"
        "  Use a large model?\n"
        "     Great answers, but slow & costly\n\n"
        "  Use a small model?\n"
        "     Fast & cheap, but are the answers\n"
        "     good enough?\n\n"
        "Existing rankings (like leaderboards) only\n"
        "measure answer quality. They ignore speed\n"
        "and cost — the factors that matter most\n"
        "when you're actually building a product."
    )
    ax.text(0.08, 0.65, problem_text, transform=ax.transAxes,
            fontsize=19, va="top", linespacing=1.5, color="#222222")

    # --- WHAT WE DID ---
    ax = fig.add_subplot(row1[1])
    ax.axis("off")
    _section_box(ax, "#fff8e1", ORANGE)
    ax.text(0.5, 0.94, "What We Did", transform=ax.transAxes,
            fontsize=32, fontweight="bold", ha="center", va="top", color=DARK)

    steps = [
        ("1", "Picked 3 AI models", "Small (1.1B), Medium (3.8B), Large (7B)\n— spanning a 7x size difference"),
        ("2", "Gave them 600 tasks", "300 article summaries (XSum dataset)\n300 reading comprehension questions (SQuAD)"),
        ("3", "Measured everything", "Answer quality (ROUGE, F1 scores)\nSpeed (seconds per answer)\nMemory usage (GB of GPU RAM)"),
        ("4", "Found the sweet spot", "Which model gives the best\nquality-per-second?"),
    ]
    y = 0.85
    for num, title, desc in steps:
        # Number circle
        ax.add_patch(FancyBboxPatch(
            (0.06, y - 0.025), 0.06, 0.055, boxstyle="round,pad=0.01",
            facecolor=ORANGE, edgecolor="none", transform=ax.transAxes
        ))
        ax.text(0.09, y, num, transform=ax.transAxes, fontsize=22,
                fontweight="bold", ha="center", va="center", color="white")
        ax.text(0.16, y + 0.01, title, transform=ax.transAxes, fontsize=21,
                fontweight="bold", va="center", color=DARK)
        ax.text(0.16, y - 0.055, desc, transform=ax.transAxes, fontsize=16,
                va="top", color="#444444", linespacing=1.4)
        y -= 0.22

    # --- THE MODELS ---
    ax = fig.add_subplot(row1[2])
    ax.axis("off")
    _section_box(ax, "#e8f5e9", GREEN)
    ax.text(0.5, 0.94, "The Three Models", transform=ax.transAxes,
            fontsize=32, fontweight="bold", ha="center", va="top", color=DARK)

    models_info = [
        ("TinyLlama", "1.1 Billion parameters", BLUE,
         "The lightweight option.\nFast and cheap, but can it keep up?", "2.4 GB RAM"),
        ("Phi-3 Mini", "3.8 Billion parameters", ORANGE,
         "The middle ground.\nMicrosoft's efficient instruction-tuned model.", "7.7 GB RAM"),
        ("Mistral-7B", "7.2 Billion parameters", GREEN,
         "The heavyweight.\nOur quality benchmark — but at what cost?", "13.9 GB RAM"),
    ]
    y = 0.82
    for name, params, color, desc, mem in models_info:
        ax.add_patch(FancyBboxPatch(
            (0.05, y - 0.06), 0.9, 0.17, boxstyle="round,pad=0.02",
            facecolor="white", edgecolor=color, linewidth=3, transform=ax.transAxes
        ))
        ax.text(0.10, y + 0.06, name, transform=ax.transAxes, fontsize=23,
                fontweight="bold", va="center", color=color)
        ax.text(0.92, y + 0.06, mem, transform=ax.transAxes, fontsize=16,
                va="center", ha="right", color="#666666",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color + "20", edgecolor="none"))
        ax.text(0.10, y + 0.02, params, transform=ax.transAxes, fontsize=16,
                va="center", color="#666666")
        ax.text(0.10, y - 0.03, desc, transform=ax.transAxes, fontsize=16,
                va="center", color="#333333", linespacing=1.4)
        y -= 0.27

    # ============================================================ #
    #  ROW 2: Results — the plots                                   #
    # ============================================================ #
    row2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[3], wspace=0.07)

    # --- SUMMARIZATION: Quality vs Speed ---
    ax = fig.add_subplot(row2[0])
    for i, m in enumerate(MODELS):
        ax.scatter(SUMM_LAT[i], SUMM_ROUGEL[i] * 100, s=600,
                   color=list(MODEL_COLORS.values())[i],
                   marker="o", edgecolors="black", linewidth=2, zorder=5, label=m)
    ax.plot(SUMM_LAT, [r * 100 for r in SUMM_ROUGEL], '--', color='gray', alpha=0.4, lw=2)
    ax.set_xlabel("Time per answer (seconds)", fontsize=18, labelpad=10)
    ax.set_ylabel("Summary Quality (ROUGE-L %)", fontsize=18, labelpad=10)
    ax.set_title("Summarization Task", fontsize=26, fontweight="bold", pad=20)
    ax.legend(fontsize=16, loc="lower right", markerscale=0.8)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=15)
    # Annotation
    ax.annotate("4.6x faster\nonly 10 pts lower",
                xy=(SUMM_LAT[0], SUMM_ROUGEL[0] * 100),
                xytext=(4.5, 14.5), fontsize=15, ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.5),
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#e3f2fd", edgecolor=BLUE, lw=1.5))
    # Arrow showing "ideal" direction
    ax.annotate("", xy=(1.0, 30), xytext=(3.0, 22),
                arrowprops=dict(arrowstyle="-|>", color="#aaaaaa", lw=2, ls="--"))
    ax.text(1.5, 27, "ideal", fontsize=14, color="#aaaaaa", style="italic", rotation=25)

    # --- QA: Quality vs Speed ---
    ax = fig.add_subplot(row2[1])
    for i, m in enumerate(MODELS):
        ax.scatter(QA_LAT[i], QA_F1[i] * 100, s=600,
                   color=list(MODEL_COLORS.values())[i],
                   marker="s", edgecolors="black", linewidth=2, zorder=5, label=m)
    ax.plot(QA_LAT, [f * 100 for f in QA_F1], '--', color='gray', alpha=0.4, lw=2)
    ax.set_xlabel("Time per answer (seconds)", fontsize=18, labelpad=10)
    ax.set_ylabel("Answer Accuracy (F1 %)", fontsize=18, labelpad=10)
    ax.set_title("Question Answering Task", fontsize=26, fontweight="bold", pad=20)
    ax.legend(fontsize=16, loc="lower right", markerscale=0.8)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=15)
    ax.annotate("TinyLlama can't\nfollow instructions!",
                xy=(QA_LAT[0], QA_F1[0] * 100),
                xytext=(2.2, 30), fontsize=15, ha="center", fontweight="bold",
                color="#c62828",
                arrowprops=dict(arrowstyle="-|>", color="#c62828", lw=2.5),
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffebee", edgecolor="#c62828", lw=1.5))

    # --- Speed & Memory comparison ---
    ax = fig.add_subplot(row2[2])
    x = np.arange(3)
    bar_w = 0.35
    colors = [BLUE, ORANGE, GREEN]

    # Throughput bars
    bars1 = ax.bar(x - bar_w/2, THROUGHPUT, bar_w, color=colors,
                   edgecolor="black", linewidth=1, label="Tokens/sec", alpha=0.85)
    for bar, v in zip(bars1, THROUGHPUT):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.0f}", ha="center", va="bottom", fontsize=16, fontweight="bold")

    ax2 = ax.twinx()
    bars2 = ax2.bar(x + bar_w/2, MEMORY_GB, bar_w, color=colors,
                    edgecolor="black", linewidth=1, alpha=0.4, hatch="//")
    for bar, v in zip(bars2, MEMORY_GB):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f"{v:.1f} GB", ha="center", va="bottom", fontsize=14, color="#666666")

    ax.set_xticks(x)
    ax.set_xticklabels(["TinyLlama\n1.1B", "Phi-3\n3.8B", "Mistral\n7B"], fontsize=15)
    ax.set_ylabel("Speed (tokens/sec, solid)", fontsize=16, labelpad=10)
    ax2.set_ylabel("Memory (GB, hatched)", fontsize=16, labelpad=10)
    ax.set_title("Speed vs. Memory Cost", fontsize=26, fontweight="bold", pad=20)
    ax.tick_params(labelsize=14)
    ax2.tick_params(labelsize=14)
    ax.grid(axis="y", alpha=0.2)

    # ============================================================ #
    #  ROW 3: Three Takeaways                                       #
    # ============================================================ #
    row3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[4], wspace=0.05)

    # --- TAKEAWAY 1 ---
    ax = fig.add_subplot(row3[0])
    ax.axis("off")
    _section_box(ax, "#e3f2fd", BLUE)
    ax.text(0.5, 0.92, "Takeaway #1", transform=ax.transAxes,
            fontsize=24, fontweight="bold", ha="center", va="top", color="#666666")
    ax.text(0.5, 0.72, "Phi-3 is the\nsweet spot", transform=ax.transAxes,
            fontsize=34, fontweight="bold", ha="center", va="center", color=DARK,
            linespacing=1.3)
    ax.add_patch(FancyBboxPatch(
        (0.08, 0.20), 0.84, 0.30, boxstyle="round,pad=0.03",
        facecolor="white", edgecolor=BLUE, linewidth=2, transform=ax.transAxes
    ))
    ax.text(0.5, 0.38, "89% of the big model's quality\nat 45% of the speed cost",
            transform=ax.transAxes, fontsize=21, ha="center", va="center",
            color=DARK, linespacing=1.5, fontweight="bold")
    ax.text(0.5, 0.10, "Best quality-per-second of all three models.",
            transform=ax.transAxes, fontsize=18, ha="center", va="center",
            color="#666666", style="italic")

    # --- TAKEAWAY 2 ---
    ax = fig.add_subplot(row3[1])
    ax.axis("off")
    _section_box(ax, "#fff8e1", ORANGE)
    ax.text(0.5, 0.92, "Takeaway #2", transform=ax.transAxes,
            fontsize=24, fontweight="bold", ha="center", va="top", color="#666666")
    ax.text(0.5, 0.72, "The task\nmatters a lot", transform=ax.transAxes,
            fontsize=34, fontweight="bold", ha="center", va="center", color=DARK,
            linespacing=1.3)
    ax.add_patch(FancyBboxPatch(
        (0.08, 0.20), 0.84, 0.30, boxstyle="round,pad=0.03",
        facecolor="white", edgecolor=ORANGE, linewidth=2, transform=ax.transAxes
    ))
    ax.text(0.5, 0.42, "Summarization gap:  10 points",
            transform=ax.transAxes, fontsize=20, ha="center", va="center",
            color=GREEN, fontweight="bold")
    ax.text(0.5, 0.32, "Question answering gap:  39 points",
            transform=ax.transAxes, fontsize=20, ha="center", va="center",
            color="#c62828", fontweight="bold")
    ax.text(0.5, 0.10, "Harder tasks expose bigger quality gaps\nbetween small and large models.",
            transform=ax.transAxes, fontsize=18, ha="center", va="center",
            color="#666666", style="italic", linespacing=1.4)

    # --- TAKEAWAY 3 ---
    ax = fig.add_subplot(row3[2])
    ax.axis("off")
    _section_box(ax, "#e8f5e9", GREEN)
    ax.text(0.5, 0.92, "Takeaway #3", transform=ax.transAxes,
            fontsize=24, fontweight="bold", ha="center", va="top", color="#666666")
    ax.text(0.5, 0.72, "Small models\ndon't listen", transform=ax.transAxes,
            fontsize=34, fontweight="bold", ha="center", va="center", color=DARK,
            linespacing=1.3)
    ax.add_patch(FancyBboxPatch(
        (0.08, 0.20), 0.84, 0.30, boxstyle="round,pad=0.03",
        facecolor="white", edgecolor=GREEN, linewidth=2, transform=ax.transAxes
    ))
    ax.text(0.5, 0.42, 'Q: "When was the Eiffel Tower built?"', transform=ax.transAxes,
            fontsize=18, ha="center", va="center", color="#666666")
    ax.text(0.5, 0.35, 'Mistral: "1889"     (correct)',
            transform=ax.transAxes, fontsize=18, ha="center", va="center",
            color=GREEN, fontweight="bold")
    ax.text(0.5, 0.28, 'TinyLlama: "The Eiffel Tower is a\nfamous landmark..."     (missed it)',
            transform=ax.transAxes, fontsize=18, ha="center", va="center",
            color="#c62828", fontweight="bold", linespacing=1.3)
    ax.text(0.5, 0.10, "Smaller models often ignore instructions\nand ramble instead of answering.",
            transform=ax.transAxes, fontsize=18, ha="center", va="center",
            color="#666666", style="italic", linespacing=1.4)

    fig.savefig(OUTPUT_PATH, dpi=100, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Poster saved to: {OUTPUT_PATH}")


def _section_box(ax, facecolor, edgecolor):
    ax.add_patch(FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0.02",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=3, transform=ax.transAxes, zorder=0
    ))


if __name__ == "__main__":
    main()
