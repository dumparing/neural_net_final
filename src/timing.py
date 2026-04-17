"""
timing.py – Latency and throughput measurement utilities.

Timing methodology:
  • Each model is warmed up with a few dummy generations before any
    measured run.  This ensures caches (KV-cache, CUDA kernels) are
    primed and the first real example is not penalized.
  • For each example we record wall-clock time around the generate call,
    count the tokens produced, and derive tokens-per-second.
  • We deliberately avoid torch.cuda.Event timing because it is not
    portable to CPU / MPS.  Wall-clock time with time.perf_counter() is
    used instead, which is monotonic and high-resolution on all platforms.
  • Optional peak-memory tracking is CUDA-only and silently skipped on
    other devices.
"""

import time
from typing import Any, Dict, Optional

import torch

from src.models import generate_text
from src.utils import log


# ================================================================== #
#  Warmup                                                              #
# ================================================================== #

def warmup(
    model,
    tokenizer,
    device: torch.device,
    steps: int = 3,
    max_new_tokens: int = 32,
) -> None:
    """
    Run a few throwaway generations to warm up the model.
    Uses a trivial prompt so this is quick.
    """
    log(f"Warming up model ({steps} steps) ...")
    dummy_prompt = "Hello, how are you?"
    for _ in range(steps):
        generate_text(
            model, tokenizer, dummy_prompt, device,
            max_new_tokens=max_new_tokens,
        )
    # Sync CUDA if available
    if device.type == "cuda":
        torch.cuda.synchronize()
    log("  → Warmup complete.")


# ================================================================== #
#  Timed generation for a single example                               #
# ================================================================== #

def timed_generate(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
    repetition_penalty: float = 1.0,
    measure_memory: bool = False,
) -> Dict[str, Any]:
    """
    Generate text and return a dict with the output plus timing info.

    Returned dict keys:
      - output_text: the generated string
      - latency_s: end-to-end wall-clock seconds
      - num_tokens: number of new tokens generated
      - tokens_per_sec: throughput
      - peak_memory_mb: (optional) CUDA peak memory, or None
    """
    peak_memory_mb: Optional[float] = None

    # Reset peak memory tracker before this generation
    if measure_memory and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Synchronize before timing to avoid measuring leftover GPU work
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    output_text = generate_text(
        model, tokenizer, prompt, device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
    )

    # Synchronize after to ensure all GPU work is done
    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    # Count tokens in the generated output
    num_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))

    tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0.0

    # Peak memory (CUDA only)
    if measure_memory and device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "output_text": output_text,
        "latency_s": round(elapsed, 4),
        "num_tokens": num_tokens,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "peak_memory_mb": round(peak_memory_mb, 1) if peak_memory_mb is not None else None,
    }
