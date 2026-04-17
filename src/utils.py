"""
utils.py – Shared helpers: seeding, device resolution, safe I/O, logging.
"""

import json
import os
import random
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch


# ------------------------------------------------------------------ #
#  Reproducibility                                                     #
# ------------------------------------------------------------------ #

def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------ #
#  Device / dtype resolution                                           #
# ------------------------------------------------------------------ #

def resolve_device(device_str: str) -> torch.device:
    """
    Convert a config string like 'auto', 'cpu', 'cuda', 'mps'
    into a torch.device, falling back gracefully.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def resolve_dtype(dtype_str: str, device: torch.device) -> torch.dtype:
    """
    Convert a config string like 'auto', 'float16', 'bfloat16', 'float32'
    into a torch.dtype.  'auto' picks the best option for the device.
    """
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_str != "auto":
        return mapping.get(dtype_str, torch.float32)

    # Auto: prefer float16 on CUDA, float32 on CPU/MPS
    if device.type == "cuda":
        # Use bfloat16 if the GPU supports it (Ampere+), else float16
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


# ------------------------------------------------------------------ #
#  File I/O helpers                                                    #
# ------------------------------------------------------------------ #

def save_json(data: Any, path: str) -> None:
    """Write *data* as pretty-printed JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log(f"Saved {path}")


def load_json(path: str) -> Any:
    """Load a JSON file if it exists, else return None."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def save_jsonl(records: List[Dict], path: str) -> None:
    """Write a list of dicts as JSON-Lines."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")
    log(f"Saved {path} ({len(records)} records)")


# ------------------------------------------------------------------ #
#  Logging                                                             #
# ------------------------------------------------------------------ #

def log(msg: str) -> None:
    """Simple timestamped log to stderr so stdout stays clean."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------ #
#  Model name → safe filename                                          #
# ------------------------------------------------------------------ #

def model_name_to_id(name: str) -> str:
    """
    Convert 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' → 'tinyllama-1.1b-chat-v1.0'
    for use in file paths.
    """
    # Take the part after the last slash, lowercase, replace unsafe chars
    short = name.split("/")[-1].lower()
    safe = short.replace(" ", "-")
    return safe


def result_exists(output_dir: str, model_id: str, task: str) -> bool:
    """Check whether aggregate metrics already exist for a model/task combo."""
    path = os.path.join(output_dir, model_id, task, "aggregate_metrics.json")
    return os.path.exists(path)
