"""
config.py – Load YAML configuration and merge with CLI overrides.

The configuration flows as:
  1. Load defaults from configs/default.yaml
  2. Override with any CLI arguments the user supplied
  3. Apply dry-run adjustments if enabled
"""

import argparse
import os
import yaml
from typing import Any, Dict


# ------------------------------------------------------------------ #
#  YAML loading                                                       #
# ------------------------------------------------------------------ #

def load_yaml_config(path: str) -> Dict[str, Any]:
    """Read a YAML file and return its contents as a dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg if cfg is not None else {}


# ------------------------------------------------------------------ #
#  CLI argument parser                                                 #
# ------------------------------------------------------------------ #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Quality–Latency Tradeoff Evaluation for Instruction-Tuned LMs"
    )
    p.add_argument("--config", type=str, default="configs/default.yaml",
                    help="Path to YAML config file")
    p.add_argument("--models", nargs="+", default=None,
                    help="Override model list")
    p.add_argument("--tasks", nargs="+", default=None,
                    choices=["summarization", "qa"],
                    help="Tasks to run")
    p.add_argument("--num-samples-summarization", type=int, default=None)
    p.add_argument("--num-samples-qa", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", default=None,
                    help="Quick test with a handful of examples")
    p.add_argument("--dry-run-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-input-length", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--skip-plots", action="store_true", default=None)
    p.add_argument("--skip-memory-measurement", action="store_true", default=None)
    p.add_argument("--resume", action="store_true", default=None,
                    help="Skip model/task combos whose results already exist")
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    return p


# ------------------------------------------------------------------ #
#  Merge YAML defaults + CLI overrides                                 #
# ------------------------------------------------------------------ #

def get_config() -> Dict[str, Any]:
    """Return the final merged configuration dict."""
    parser = build_parser()
    args = parser.parse_args()

    # 1. Load YAML defaults
    cfg = load_yaml_config(args.config)

    # 2. CLI overrides – only set keys the user explicitly provided
    cli_map = {
        "models": args.models,
        "tasks": args.tasks,
        "num_samples_summarization": args.num_samples_summarization,
        "num_samples_qa": args.num_samples_qa,
        "dry_run": args.dry_run,
        "dry_run_samples": args.dry_run_samples,
        "seed": args.seed,
        "max_input_length": args.max_input_length,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "device": args.device,
        "dtype": args.dtype,
        "output_dir": args.output_dir,
        "skip_plots": args.skip_plots,
        "skip_memory_measurement": args.skip_memory_measurement,
        "resume": args.resume,
        "warmup_steps": args.warmup_steps,
        "batch_size": args.batch_size,
    }
    for key, val in cli_map.items():
        if val is not None:
            cfg[key] = val

    # 3. Dry-run adjustments
    if cfg.get("dry_run", False):
        n = cfg.get("dry_run_samples", 10)
        cfg["num_samples_summarization"] = n
        cfg["num_samples_qa"] = n
        # Also reduce warmup for speed
        cfg["warmup_steps"] = min(cfg.get("warmup_steps", 3), 1)

    # 4. Ensure output directory exists
    os.makedirs(cfg.get("output_dir", "outputs"), exist_ok=True)

    return cfg
