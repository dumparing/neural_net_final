# When Is Smaller Better? Evaluating Quality–Latency Tradeoffs in Instruction-Tuned Language Models

A reproducible experimental pipeline that compares smaller and larger instruction-tuned language models on both output quality and inference efficiency.

## Research Question

> When is a smaller language model "good enough" compared to a larger one, considering both output quality and latency?

## Models

| Model | Parameters |
|---|---|
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | ~1.1B |
| microsoft/Phi-3-mini-4k-instruct | ~3.8B |
| mistralai/Mistral-7B-Instruct-v0.3 | ~7.2B |

## Tasks

- **Summarization** — XSum dataset, evaluated with ROUGE-1/2/L
- **Question Answering** — SQuAD v1.1, evaluated with Exact Match & token F1

## Project Structure

```
├── main.py                 # Entry point
├── configs/
│   └── default.yaml        # All tunable parameters
├── src/
│   ├── config.py           # YAML + CLI argument merging
│   ├── data.py             # Dataset loading (XSum, SQuAD)
│   ├── models.py           # Model loading, generation, cleanup
│   ├── prompts.py          # Prompt templates & chat formatting
│   ├── metrics.py          # ROUGE, EM, F1 scoring
│   ├── timing.py           # Warmup & timed generation
│   ├── experiments.py      # Evaluation loops
│   ├── plotting.py         # Matplotlib plots
│   ├── pareto.py           # Pareto frontier analysis
│   └── utils.py            # Seeding, I/O, device resolution
├── outputs/                # All results go here
├── requirements.txt
└── README.md
```

## Installation

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) If you have a CUDA GPU, install the matching PyTorch build:
#    https://pytorch.org/get-started/locally/
```

## Quick Start

### Dry run (10 examples, ~5 minutes on GPU)

```bash
python main.py --dry-run
```

### Summarization only

```bash
python main.py --tasks summarization
```

### QA only

```bash
python main.py --tasks qa
```

### Full experiment (300 examples per task, all 3 models)

```bash
python main.py
```

### Resume an interrupted run

```bash
python main.py --resume
```

### Single model test

```bash
python main.py --dry-run --models "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

### Custom sample size

```bash
python main.py --num-samples-summarization 100 --num-samples-qa 100
```

### Skip plots

```bash
python main.py --skip-plots
```

## Configuration

All parameters can be set in `configs/default.yaml` or overridden on the command line. Key options:

| Parameter | Default | Description |
|---|---|---|
| `--dry-run` | false | Use ~10 examples for quick testing |
| `--models` | all 3 | Space-separated model list |
| `--tasks` | both | `summarization`, `qa`, or both |
| `--num-samples-summarization` | 300 | Number of XSum examples |
| `--num-samples-qa` | 300 | Number of SQuAD examples |
| `--max-new-tokens` | 256 | Max tokens to generate |
| `--temperature` | 0.0 | 0 = greedy decoding |
| `--device` | auto | `auto`, `cpu`, `cuda`, `mps` |
| `--dtype` | auto | `auto`, `float16`, `bfloat16`, `float32` |
| `--output-dir` | outputs | Where results are saved |
| `--resume` | false | Skip model/task combos with existing results |
| `--skip-plots` | false | Skip plot generation |
| `--seed` | 42 | Random seed |

## Outputs

After a run, `outputs/` will contain:

```
outputs/
├── config_used.json              # Exact config for this run
├── combined_summary.json         # All aggregate metrics
├── combined_summary.csv          # Same as above, CSV format
├── pareto/
│   ├── summarization_pareto.json
│   ├── qa_pareto.json
│   └── all_pareto.json
├── plots/
│   ├── summarization_quality_vs_latency.png
│   ├── qa_quality_vs_latency.png
│   ├── summarization_comparison.png
│   └── qa_comparison.png
└── <model_id>/
    └── <task>/
        ├── predictions.jsonl         # Per-example outputs
        └── aggregate_metrics.json    # Aggregate scores + timing
```

## Timing Methodology

- Each model is **warmed up** with 3 dummy generations before any measured run.
- Latency is measured with `time.perf_counter()` (monotonic, high-resolution) around each `model.generate()` call.
- On CUDA, `torch.cuda.synchronize()` is called before and after generation to ensure accurate timing.
- Tokens-per-second is computed as `(output tokens) / (wall-clock seconds)`.
- Peak memory is tracked via `torch.cuda.max_memory_allocated()` (CUDA only; silently skipped on CPU/MPS).

## Hardware Notes

- **TinyLlama (1.1B)**: Runs comfortably on CPU or any GPU. ~2.5 GB in float16.
- **Phi-3-mini (3.8B)**: Needs ~8 GB VRAM in float16, or runs on CPU (slower).
- **Mistral-7B**: Needs ~14–15 GB VRAM in float16. Will be very slow on CPU. If you have <16 GB VRAM, consider running only TinyLlama and Phi-3, or reducing `--max-new-tokens`.
- Models are loaded **one at a time** and explicitly unloaded between runs, so you never need memory for two models simultaneously.
- On Apple Silicon (MPS), float32 is used automatically as float16 support is limited.
- All generation is done **example-by-example** (batch size 1) for simplicity and consistent timing.

## Reproducibility

- Fixed random seed (default 42) for dataset sampling and PyTorch.
- Greedy decoding (temperature=0) eliminates sampling randomness.
- The exact config used is saved to `outputs/config_used.json`.
- All per-example predictions and scores are saved.
