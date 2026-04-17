"""
data.py – Load and sample datasets (XSum for summarization, SQuAD v1.1 for QA).

Datasets are downloaded via Hugging Face `datasets` and cached locally.
A configurable number of examples is sampled with a fixed seed for
reproducibility.
"""

from typing import Any, Dict, List

from datasets import load_dataset

from src.utils import log, set_seed


# ------------------------------------------------------------------ #
#  Summarization – XSum                                                #
# ------------------------------------------------------------------ #

def load_xsum(num_samples: int, seed: int) -> List[Dict[str, Any]]:
    """
    Load the XSum test split and return *num_samples* examples.
    Each example dict has keys: 'document', 'summary', 'id'.
    """
    log(f"Loading XSum test split (sampling {num_samples} examples) ...")
    ds = load_dataset("EdinburghNLP/xsum", split="test", trust_remote_code=True)

    # Deterministic shuffle + slice
    set_seed(seed)
    ds = ds.shuffle(seed=seed)
    ds = ds.select(range(min(num_samples, len(ds))))

    examples = []
    for row in ds:
        examples.append({
            "id": row["id"],
            "document": row["document"],
            "summary": row["summary"],  # reference summary
        })
    log(f"  → {len(examples)} XSum examples ready.")
    return examples


# ------------------------------------------------------------------ #
#  Question answering – SQuAD v1.1                                     #
# ------------------------------------------------------------------ #

def load_squad(num_samples: int, seed: int) -> List[Dict[str, Any]]:
    """
    Load the SQuAD v1.1 validation split and return *num_samples* examples.
    Each example dict has keys: 'id', 'context', 'question', 'answers'.
    """
    log(f"Loading SQuAD v1.1 validation split (sampling {num_samples} examples) ...")
    ds = load_dataset("rajpurkar/squad", split="validation", trust_remote_code=True)

    set_seed(seed)
    ds = ds.shuffle(seed=seed)
    ds = ds.select(range(min(num_samples, len(ds))))

    examples = []
    for row in ds:
        examples.append({
            "id": row["id"],
            "context": row["context"],
            "question": row["question"],
            "answers": row["answers"]["text"],  # list of gold answer strings
        })
    log(f"  → {len(examples)} SQuAD examples ready.")
    return examples
