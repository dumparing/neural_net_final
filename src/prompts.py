"""
prompts.py – Prompt templates and chat-formatting utilities.

Design:
  • One canonical instruction per task, shared across all models.
  • If the tokenizer has a chat template, we wrap the instruction in a
    single-turn user message and call apply_chat_template().
  • If not (or if it fails), we fall back to a plain "### Instruction: …"
    format so the model still receives the same content.
"""

from typing import Dict, List, Optional

from src.utils import log


# ================================================================== #
#  Raw instruction templates                                           #
# ================================================================== #

# {document} is replaced at runtime
SUMMARIZATION_INSTRUCTION = (
    "Summarize the following article in one or two sentences.\n\n"
    "Article:\n{document}\n\n"
    "Summary:"
)

# {context} and {question} are replaced at runtime
QA_INSTRUCTION = (
    "Read the passage below and answer the question with a short, "
    "exact phrase from the passage. Do not explain.\n\n"
    "Passage:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


# ================================================================== #
#  Build a single chat-formatted (or plain) prompt string              #
# ================================================================== #

def build_prompt(
    tokenizer,
    task: str,
    example: Dict,
    max_input_length: int,
) -> str:
    """
    Return the final prompt string ready to be tokenized.

    Steps:
      1. Fill in the template with the example fields.
      2. Try to apply the tokenizer's chat template.
      3. Fall back to a plain instruction prompt if chat templating fails.
    """
    # --- Step 1: fill template -------------------------------------------
    if task == "summarization":
        instruction = SUMMARIZATION_INSTRUCTION.format(document=example["document"])
    elif task == "qa":
        instruction = QA_INSTRUCTION.format(
            context=example["context"],
            question=example["question"],
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    # --- Step 2: try chat template ---------------------------------------
    prompt = _try_chat_template(tokenizer, instruction)

    # --- Step 3: truncate to stay within budget --------------------------
    # We do a rough character-level trim first, then verify with the
    # tokenizer.  This avoids accidentally blowing past max_input_length.
    prompt = _truncate_prompt(tokenizer, prompt, max_input_length)

    return prompt


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #

def _try_chat_template(tokenizer, instruction: str) -> str:
    """
    Attempt to wrap *instruction* in the tokenizer's chat template.
    Returns the formatted string, or a plain fallback.
    """
    messages: List[Dict[str, str]] = [{"role": "user", "content": instruction}]
    try:
        # add_generation_prompt=True appends the assistant turn prefix
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return formatted
    except Exception:
        # Fallback: plain instruction format
        return f"### Instruction:\n{instruction}\n\n### Response:\n"


def _truncate_prompt(tokenizer, prompt: str, max_tokens: int) -> str:
    """
    If the prompt exceeds *max_tokens*, truncate from the middle of the
    document/passage to keep instruction framing intact.

    As a simple approach we just hard-truncate the token sequence and
    decode back.  The generation call will work on the truncated version.
    """
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return prompt
    # Keep the first max_tokens tokens
    truncated_ids = ids[:max_tokens]
    return tokenizer.decode(truncated_ids, skip_special_tokens=False)
