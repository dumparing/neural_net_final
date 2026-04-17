"""
pareto.py – Pareto frontier analysis.

For each task we identify which models are Pareto-efficient:
  • Higher quality is better  (maximise)
  • Lower latency is better   (minimise)

A point is Pareto-efficient if no other point is simultaneously
better on both objectives.
"""

import os
from typing import Any, Dict, List

from src.utils import log, save_json


# ================================================================== #
#  Core Pareto computation                                             #
# ================================================================== #

def is_pareto_efficient(points: List[Dict[str, float]]) -> List[bool]:
    """
    Given a list of dicts with keys 'quality' and 'latency',
    return a boolean mask indicating which points are on the Pareto frontier.

    Convention:
      - higher quality is better
      - lower latency is better
    A point p dominates q iff p.quality >= q.quality AND p.latency <= q.latency
    with at least one strict inequality.
    """
    n = len(points)
    efficient = [True] * n

    for i in range(n):
        if not efficient[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            # Does j dominate i?
            if (points[j]["quality"] >= points[i]["quality"] and
                points[j]["latency"] <= points[i]["latency"] and
                (points[j]["quality"] > points[i]["quality"] or
                 points[j]["latency"] < points[i]["latency"])):
                efficient[i] = False
                break

    return efficient


# ================================================================== #
#  Run Pareto analysis across all summaries                            #
# ================================================================== #

def run_pareto_analysis(
    summaries: List[Dict[str, Any]],
    output_dir: str,
) -> List[Dict[str, Any]]:
    """
    Compute Pareto frontiers per task.
    Saves results and returns a combined list of annotated entries.
    """
    pareto_dir = os.path.join(output_dir, "pareto")
    os.makedirs(pareto_dir, exist_ok=True)

    tasks = sorted(set(s["task"] for s in summaries))
    all_results: List[Dict[str, Any]] = []

    for task in tasks:
        task_summaries = [s for s in summaries if s["task"] == task]
        if not task_summaries:
            continue

        # Pick the primary quality metric per task
        if task == "summarization":
            quality_key = "rougeL"
        elif task == "qa":
            quality_key = "f1"
        else:
            quality_key = "f1"  # default

        # Build points list
        points = []
        for s in task_summaries:
            points.append({
                "quality": s.get(quality_key, 0.0),
                "latency": s.get("mean_latency_s", 999.0),
            })

        mask = is_pareto_efficient(points)

        # Build annotated results
        task_results = []
        for s, eff in zip(task_summaries, mask):
            entry = {
                "model": s.get("model", ""),
                "model_id": s.get("model_id", ""),
                "task": task,
                "quality_metric": quality_key,
                "quality_value": s.get(quality_key, 0.0),
                "mean_latency_s": s.get("mean_latency_s", 0.0),
                "pareto_efficient": eff,
            }
            task_results.append(entry)
            all_results.append(entry)

        # Save per-task
        save_json(task_results, os.path.join(pareto_dir, f"{task}_pareto.json"))

        # Print readable summary
        log(f"\nPareto frontier – {task} (quality={quality_key}, latency=mean_latency_s):")
        log(f"  {'Model':<45} {'Quality':>8} {'Latency':>10} {'Pareto?':>8}")
        log(f"  {'-'*45} {'-'*8} {'-'*10} {'-'*8}")
        for r in task_results:
            flag = " ★" if r["pareto_efficient"] else ""
            log(f"  {r['model_id']:<45} {r['quality_value']:>8.4f} "
                f"{r['mean_latency_s']:>9.3f}s {flag:>8}")

    # Save combined
    save_json(all_results, os.path.join(pareto_dir, "all_pareto.json"))

    return all_results
