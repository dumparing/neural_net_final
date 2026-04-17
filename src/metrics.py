"""
metrics.py – Quality evaluation for summarization and QA.

Summarization: ROUGE-1, ROUGE-2, ROUGE-L  (via the `evaluate` library)
QA:            Exact Match and token-level F1 (custom, SQuAD-style)

All metrics operate on plain strings and return dicts.
"""

import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import evaluate

from src.utils import log


# ================================================================== #
#  Summarization – ROUGE                                               #
# ================================================================== #

# Load the metric once (lazy global)
_rouge_metric = None

def _get_rouge():
    global _rouge_metric
    if _rouge_metric is None:
        _rouge_metric = evaluate.load("rouge")
    return _rouge_metric


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute corpus-level ROUGE-1, ROUGE-2, ROUGE-L.
    Returns a dict with keys 'rouge1', 'rouge2', 'rougeL', each a float in [0,1].
    """
    rouge = _get_rouge()
    results = rouge.compute(predictions=predictions, references=references)
    return {
        "rouge1": round(results["rouge1"], 4),
        "rouge2": round(results["rouge2"], 4),
        "rougeL": round(results["rougeL"], 4),
    }


def compute_rouge_single(prediction: str, reference: str) -> Dict[str, float]:
    """Per-example ROUGE scores."""
    return compute_rouge([prediction], [reference])


# ================================================================== #
#  QA – Exact Match & token-level F1 (SQuAD-style)                     #
# ================================================================== #

def normalize_answer(text: str) -> str:
    """
    Standard SQuAD answer normalization:
    lowercase → remove punctuation → remove articles → strip whitespace.
    """
    text = text.lower()
    # Remove punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def exact_match(prediction: str, gold_answers: List[str]) -> float:
    """1.0 if the normalized prediction matches any gold answer, else 0.0."""
    pred_norm = normalize_answer(prediction)
    for ans in gold_answers:
        if normalize_answer(ans) == pred_norm:
            return 1.0
    return 0.0


def token_f1(prediction: str, gold_answers: List[str]) -> float:
    """
    Token-level F1 between prediction and the best-matching gold answer.
    Uses the standard SQuAD definition.
    """
    pred_tokens = normalize_answer(prediction).split()

    best_f1 = 0.0
    for ans in gold_answers:
        gold_tokens = normalize_answer(ans).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            continue
        precision = num_common / len(pred_tokens) if pred_tokens else 0
        recall = num_common / len(gold_tokens) if gold_tokens else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        best_f1 = max(best_f1, f1)
    return round(best_f1, 4)


def compute_qa_metrics_single(
    prediction: str,
    gold_answers: List[str],
) -> Dict[str, float]:
    """Per-example EM and F1."""
    # Try to extract a short answer from model output
    cleaned = extract_short_answer(prediction)
    return {
        "exact_match": exact_match(cleaned, gold_answers),
        "f1": token_f1(cleaned, gold_answers),
    }


def compute_qa_metrics_aggregate(
    predictions: List[str],
    gold_answers_list: List[List[str]],
) -> Dict[str, float]:
    """Corpus-level EM and F1 (macro-averaged)."""
    ems, f1s = [], []
    for pred, golds in zip(predictions, gold_answers_list):
        cleaned = extract_short_answer(pred)
        ems.append(exact_match(cleaned, golds))
        f1s.append(token_f1(cleaned, golds))
    return {
        "exact_match": round(sum(ems) / len(ems), 4) if ems else 0.0,
        "f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
    }


# ================================================================== #
#  Answer extraction heuristic                                         #
# ================================================================== #

def extract_short_answer(text: str) -> str:
    """
    Try to extract a concise answer from a potentially verbose model output.

    Heuristics (in order):
      1. If the output contains "Answer:" take everything after it.
      2. Take the first sentence / first line.
      3. Return the whole string stripped.
    """
    text = text.strip()

    # Heuristic 1: explicit "Answer:" marker
    for marker in ["Answer:", "answer:", "A:"]:
        if marker in text:
            text = text.split(marker, 1)[1].strip()
            break

    # Heuristic 2: take the first sentence
    for sep in ["\n", ". ", ".\n"]:
        if sep in text:
            text = text.split(sep)[0].strip()
            break

    # Remove trailing period if present
    if text.endswith("."):
        text = text[:-1].strip()

    return text


# ================================================================== #
#  Bootstrap confidence intervals                                      #
# ================================================================== #

def bootstrap_ci(
    scores: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute a bootstrap confidence interval for the mean of *scores*.

    Returns (mean, ci_lower, ci_upper).

    Method: resample with replacement *n_bootstrap* times, compute the
    mean of each resample, then take percentiles of the bootstrap
    distribution.  This is the standard percentile bootstrap, which is
    simple, assumption-free, and widely accepted in NLP evaluation.
    """
    rng = np.random.RandomState(seed)
    scores_arr = np.array(scores)
    n = len(scores_arr)

    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(scores_arr, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    mean = float(scores_arr.mean())

    return (round(mean, 4), round(ci_lower, 4), round(ci_upper, 4))
