#!/usr/bin/env python3
"""
main.py – Entry point for the quality–latency tradeoff study.

Usage examples:
    # Full experiment with defaults
    python main.py

    # Dry run (10 examples, fast)
    python main.py --dry-run

    # Summarization only
    python main.py --tasks summarization

    # QA only
    python main.py --tasks qa

    # Specific model
    python main.py --models "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Resume an interrupted run
    python main.py --resume

    # Custom config file
    python main.py --config configs/default.yaml
"""

import json
import os
import sys
import time

import pandas as pd

from src.config import get_config
from src.experiments import run_all_experiments
from src.pareto import run_pareto_analysis
from src.plotting import generate_all_plots
from src.utils import log, save_json, set_seed


def main():
    # ---- Configuration --------------------------------------------------
    cfg = get_config()
    set_seed(cfg.get("seed", 42))

    log("=" * 60)
    log("Quality–Latency Tradeoff Evaluation")
    log("=" * 60)
    log(f"Models : {cfg['models']}")
    log(f"Tasks  : {cfg['tasks']}")
    log(f"Samples: summarization={cfg['num_samples_summarization']}, "
        f"qa={cfg['num_samples_qa']}")
    log(f"Device : {cfg['device']}  |  Dtype: {cfg['dtype']}")
    log(f"Output : {cfg['output_dir']}")
    if cfg.get("dry_run"):
        log("*** DRY-RUN MODE ***")
    log("=" * 60)

    # Save config for reproducibility
    save_json(cfg, os.path.join(cfg["output_dir"], "config_used.json"))

    # ---- Run experiments ------------------------------------------------
    start_time = time.time()
    summaries = run_all_experiments(cfg)
    elapsed = time.time() - start_time

    if not summaries:
        log("No results produced.  Check your config and try again.")
        sys.exit(1)

    # ---- Save combined summary ------------------------------------------
    save_json(summaries, os.path.join(cfg["output_dir"], "combined_summary.json"))

    # Also save as CSV for easy inspection
    df = pd.DataFrame(summaries)
    csv_path = os.path.join(cfg["output_dir"], "combined_summary.csv")
    df.to_csv(csv_path, index=False)
    log(f"Saved {csv_path}")

    # Print a readable table
    log("\n" + "=" * 60)
    log("RESULTS SUMMARY")
    log("=" * 60)
    print(df.to_string(index=False))
    print()

    # ---- Pareto analysis ------------------------------------------------
    pareto_results = run_pareto_analysis(summaries, cfg["output_dir"])

    # ---- Plots ----------------------------------------------------------
    if not cfg.get("skip_plots", False):
        log("\nGenerating plots ...")
        try:
            generate_all_plots(summaries, cfg["output_dir"])
        except Exception as e:
            log(f"Warning: plot generation failed: {e}")
            log("Results are still saved.  You can re-generate plots later.")

    # ---- Done -----------------------------------------------------------
    log(f"\nTotal wall-clock time: {elapsed/60:.1f} minutes")
    log(f"All outputs saved to: {cfg['output_dir']}/")
    log("Done.")


if __name__ == "__main__":
    main()
