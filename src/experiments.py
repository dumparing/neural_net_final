"""
experiments.py – Run evaluation loops for summarization and QA.

This module orchestrates:
  1. Build prompts for every example
  2. Warm up the model
  3. Generate + time each example
  4. Score quality (ROUGE / EM+F1)
  5. Save per-example and aggregate results
  6. Return a summary dict for downstream analysis
"""

import os
from typing import Any, Dict, List, Tuple

import numpy as np

from src.data import load_xsum, load_squad
from src.metrics import (
    bootstrap_ci,
    compute_rouge,
    compute_rouge_single,
    compute_qa_metrics_single,
    compute_qa_metrics_aggregate,
)
from src.models import generate_text, load_model, unload_model
from src.prompts import build_prompt
from src.timing import timed_generate, warmup
from src.utils import (
    log,
    model_name_to_id,
    result_exists,
    save_json,
    save_jsonl,
    set_seed,
)


# ================================================================== #
#  Top-level experiment runner                                         #
# ================================================================== #

def run_all_experiments(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Run every (model, task) combination specified in *cfg*.
    Returns a list of summary dicts, one per combination.
    """
    all_summaries: List[Dict[str, Any]] = []

    for model_name in cfg["models"]:
        model_id = model_name_to_id(model_name)

        # Check if we can skip all tasks for this model (resume mode)
        tasks_to_run = []
        for task in cfg["tasks"]:
            if cfg.get("resume", False) and result_exists(cfg["output_dir"], model_id, task):
                log(f"[resume] Skipping {model_id}/{task} – results exist.")
                # Still load the existing summary so we include it in final output
                existing = _load_existing_summary(cfg["output_dir"], model_id, task)
                if existing:
                    all_summaries.append(existing)
                continue
            tasks_to_run.append(task)

        if not tasks_to_run:
            continue

        # Load model once, run all remaining tasks, then unload
        model, tokenizer, device = load_model(
            model_name,
            device_str=cfg.get("device", "auto"),
            dtype_str=cfg.get("dtype", "auto"),
        )

        for task in tasks_to_run:
            log(f"\n{'='*60}")
            log(f"  Model: {model_name}  |  Task: {task}")
            log(f"{'='*60}")

            summary = _run_single(cfg, model, tokenizer, device, model_name, task)
            all_summaries.append(summary)

        unload_model(model, tokenizer)

    return all_summaries


# ================================================================== #
#  Single model × task run                                             #
# ================================================================== #

def _run_single(
    cfg: Dict[str, Any],
    model,
    tokenizer,
    device,
    model_name: str,
    task: str,
) -> Dict[str, Any]:
    """Run evaluation for one model on one task and save results."""
    model_id = model_name_to_id(model_name)
    out_dir = os.path.join(cfg["output_dir"], model_id, task)
    os.makedirs(out_dir, exist_ok=True)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    # ---- Load data --------------------------------------------------
    if task == "summarization":
        examples = load_xsum(cfg["num_samples_summarization"], seed)
    elif task == "qa":
        examples = load_squad(cfg["num_samples_qa"], seed)
    else:
        raise ValueError(f"Unknown task: {task}")

    # ---- Warmup -----------------------------------------------------
    warmup(model, tokenizer, device, steps=cfg.get("warmup_steps", 3))

    # ---- Generate + time each example --------------------------------
    records: List[Dict[str, Any]] = []
    measure_mem = not cfg.get("skip_memory_measurement", False)

    for i, ex in enumerate(examples):
        prompt = build_prompt(tokenizer, task, ex, cfg["max_input_length"])
        result = timed_generate(
            model, tokenizer, prompt, device,
            max_new_tokens=cfg["max_new_tokens"],
            temperature=cfg.get("temperature", 0.0),
            top_p=cfg.get("top_p", 1.0),
            do_sample=cfg.get("do_sample", False),
            repetition_penalty=cfg.get("repetition_penalty", 1.0),
            measure_memory=measure_mem,
        )

        # Build per-example record
        rec: Dict[str, Any] = {
            "example_id": ex.get("id", i),
            "prediction": result["output_text"],
            "latency_s": result["latency_s"],
            "num_tokens": result["num_tokens"],
            "tokens_per_sec": result["tokens_per_sec"],
            "peak_memory_mb": result["peak_memory_mb"],
        }

        # Add references & per-example quality scores
        if task == "summarization":
            rec["reference"] = ex["summary"]
            scores = compute_rouge_single(result["output_text"], ex["summary"])
            rec.update(scores)
        elif task == "qa":
            rec["reference"] = ex["answers"]
            rec["question"] = ex["question"]
            scores = compute_qa_metrics_single(result["output_text"], ex["answers"])
            rec.update(scores)

        records.append(rec)

        # Progress log every 25 examples
        if (i + 1) % 25 == 0 or (i + 1) == len(examples):
            log(f"  [{i+1}/{len(examples)}] last latency={result['latency_s']:.2f}s")

    # ---- Save per-example results ------------------------------------
    save_jsonl(records, os.path.join(out_dir, "predictions.jsonl"))

    # ---- Compute aggregate metrics -----------------------------------
    agg = _aggregate(records, task)
    agg["model"] = model_name
    agg["model_id"] = model_id
    agg["task"] = task
    agg["num_examples"] = len(records)

    save_json(agg, os.path.join(out_dir, "aggregate_metrics.json"))
    log(f"  Aggregate metrics: {agg}")

    return agg


# ================================================================== #
#  Aggregation                                                         #
# ================================================================== #

def _aggregate(records: List[Dict[str, Any]], task: str) -> Dict[str, Any]:
    """Compute aggregate quality + efficiency metrics."""
    latencies = [r["latency_s"] for r in records]
    tps = [r["tokens_per_sec"] for r in records]
    lengths = [r["num_tokens"] for r in records]

    agg: Dict[str, Any] = {
        "mean_latency_s": round(float(np.mean(latencies)), 4),
        "median_latency_s": round(float(np.median(latencies)), 4),
        "mean_tokens_per_sec": round(float(np.mean(tps)), 2),
        "mean_output_length": round(float(np.mean(lengths)), 1),
    }

    # Quality metrics + bootstrap 95% confidence intervals
    if task == "summarization":
        preds = [r["prediction"] for r in records]
        refs = [r["reference"] for r in records]
        rouge = compute_rouge(preds, refs)
        agg.update(rouge)

        # Per-example ROUGE scores for bootstrap CIs
        r1_scores = [r["rouge1"] for r in records]
        r2_scores = [r["rouge2"] for r in records]
        rl_scores = [r["rougeL"] for r in records]
        for key, scores in [("rouge1", r1_scores), ("rouge2", r2_scores), ("rougeL", rl_scores)]:
            mean, lo, hi = bootstrap_ci(scores)
            agg[f"{key}_ci_lower"] = lo
            agg[f"{key}_ci_upper"] = hi

    elif task == "qa":
        preds = [r["prediction"] for r in records]
        golds = [r["reference"] for r in records]
        qa_scores = compute_qa_metrics_aggregate(preds, golds)
        agg.update(qa_scores)

        # Per-example scores for bootstrap CIs
        em_scores = [r["exact_match"] for r in records]
        f1_scores = [r["f1"] for r in records]
        for key, scores in [("exact_match", em_scores), ("f1", f1_scores)]:
            mean, lo, hi = bootstrap_ci(scores)
            agg[f"{key}_ci_lower"] = lo
            agg[f"{key}_ci_upper"] = hi

    # Bootstrap CI for latency too
    lat_mean, lat_lo, lat_hi = bootstrap_ci(latencies)
    agg["mean_latency_ci_lower"] = lat_lo
    agg["mean_latency_ci_upper"] = lat_hi

    # Memory (if available)
    mems = [r["peak_memory_mb"] for r in records if r["peak_memory_mb"] is not None]
    if mems:
        agg["peak_memory_mb"] = round(float(max(mems)), 1)

    return agg


# ================================================================== #
#  Resume helper                                                       #
# ================================================================== #

def _load_existing_summary(output_dir: str, model_id: str, task: str):
    """Try to load an existing aggregate_metrics.json for resume mode."""
    from src.utils import load_json
    path = os.path.join(output_dir, model_id, task, "aggregate_metrics.json")
    return load_json(path)
