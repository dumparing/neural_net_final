"""
models.py – Model loading, generation, and cleanup.

Each model is loaded one at a time to minimize VRAM usage.  After all
tasks for a model are done the caller should invoke `unload_model()` to
free GPU memory before the next model is loaded.
"""

import gc
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import log, resolve_device, resolve_dtype


# ================================================================== #
#  Load / unload                                                       #
# ================================================================== #

def load_model(
    model_name: str,
    device_str: str = "auto",
    dtype_str: str = "auto",
) -> Tuple[Any, Any, torch.device]:
    """
    Load a Hugging Face causal LM and its tokenizer.

    Returns (model, tokenizer, device).
    """
    device = resolve_device(device_str)
    dtype = resolve_dtype(dtype_str, device)

    log(f"Loading tokenizer for {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",  # important for batch generation
    )

    # Ensure pad token is set (many models lack one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    log(f"Loading model {model_name}  (device={device}, dtype={dtype}) ...")

    # For CPU we always use float32 regardless of config
    load_dtype = torch.float32 if device.type == "cpu" else dtype

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=load_dtype,
        device_map=device.type if device.type in ("cuda", "cpu") else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # If device_map wasn't used, move manually
    if device.type not in ("cuda", "cpu"):
        model = model.to(device)

    model.eval()
    log(f"  → Model loaded. Parameters: {_count_params(model):.1f}M")
    return model, tokenizer, device


def unload_model(model, tokenizer) -> None:
    """Delete model and tokenizer and free GPU memory."""
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log("Model unloaded, memory freed.")


# ================================================================== #
#  Generation                                                          #
# ================================================================== #

@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
    repetition_penalty: float = 1.0,
) -> str:
    """
    Generate text from *prompt* and return the decoded output (new tokens
    only, excluding the prompt).

    Uses greedy decoding when temperature == 0.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,  # safety cap
    ).to(device)

    input_length = inputs["input_ids"].shape[1]

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
    }

    # Greedy when temperature is 0 or very small
    if temperature > 1e-6 and do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    else:
        # Force greedy – override do_sample to False just in case
        gen_kwargs["do_sample"] = False

    output_ids = model.generate(**inputs, **gen_kwargs)

    # Decode only the new tokens
    new_ids = output_ids[0, input_length:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _count_params(model) -> float:
    """Return parameter count in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6
