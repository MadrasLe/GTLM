<div align="center">

# GTLM — Generative Transformer Language Model

**A Sparse Mixture-of-Experts language model trained from scratch on a single GPU.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org/)
[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-0.14%2B-7B68EE.svg)](https://www.deepspeed.ai/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow.svg)](https://huggingface.co/Madras1)

---

*An experimental, low-budget MoE research project demonstrating that competitive*
*small-model performance is achievable with sparse architectures and careful data curation.*

</div>

---

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
  - [Why Mixture-of-Experts?](#why-mixture-of-experts)
  - [Architecture Diagram](#architecture-diagram)
  - [Core Components](#core-components)
- [Released Checkpoints](#released-checkpoints)
- [Benchmark Results](#benchmark-results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Installation](#installation)
- [Training](#training)
  - [Dataset Format](#dataset-format)
  - [Launch Commands](#launch-commands)
  - [Configuration](#configuration)
  - [Resuming from Checkpoint](#resuming-from-checkpoint)
- [Using the Model for Inference](#using-the-model-for-inference)
- [Engineering Checks](#engineering-checks)
- [Documentation](#documentation)
- [License](#license)

---

## Overview

GTLM is a **Sparse Mixture-of-Experts (MoE) causal language model** designed and trained as a research experiment by [Gabriel Yogi (Madras1)](https://huggingface.co/Madras1). The flagship release, **GTLM-1-2B-A350M**, packs **~2 billion total parameters** but activates only **~350 million per token** through top-2 expert routing — achieving parameter efficiency comparable to models 3–5× smaller while retaining the representational capacity of a much larger network.

### Key Highlights

| | |
|---|---|
| 🧠 **Architecture** | Decoder-only Transformer with Sparse MoE, RoPE, SwiGLU experts, RMSNorm |
| ⚡ **Efficiency** | Only ~17% of parameters active per token via top-2 routing |
| 💰 **Training cost** | ~100 USD on a single NVIDIA A100 (~140 hours) |
| 📊 **Data** | 15B tokens — curated English + Brazilian Portuguese blend |
| 🔧 **Stack** | PyTorch, DeepSpeed, Flash Attention 2, Liger Kernels |
| 📈 **Performance** | 56.2% average on zero-shot benchmarks — competitive with dense models trained on 20× more data |

---

## Model Architecture

### Why Mixture-of-Experts?

Traditional (dense) transformers activate **every parameter** for every input token. This is wasteful: most of the network's capacity is not needed for any given input. **Mixture-of-Experts** solves this by replacing the standard feed-forward block with a collection of smaller "expert" networks and a lightweight **router** that selects which experts process each token.

The result: a model that has the **knowledge capacity** of a large network but the **computational cost** of a small one.

```
Dense model:     Every token → ALL parameters    → expensive
MoE model:       Every token → SELECTED experts   → efficient
```

In GTLM-1, each layer has **16 expert MLPs**, and the router picks the **top 2** for each token. This means only ~350M out of ~2B parameters are active at any given time.

### Architecture Diagram

```
Input Tokens
     │
     ▼
┌─────────────────┐
│  Token Embedding │   (vocab_size × hidden_dim)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Dropout      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         × N Transformer Blocks          │
│                                         │
│   ┌───────────┐    ┌─────────────────┐  │
│   │  RMSNorm  │───▶│   Multi-Head    │  │
│   └───────────┘    │   Attention     │  │
│        │           │   + RoPE        │  │
│        │           │   + Flash Attn  │  │
│        │           └────────┬────────┘  │
│        └──── + residual ────┘           │
│                     │                   │
│   ┌───────────┐    ┌─────────────────┐  │
│   │  RMSNorm  │───▶│   MoE Layer     │  │
│   └───────────┘    │   (Top-K Gate)  │  │
│        │           │   ┌──┐┌──┐┌──┐  │  │
│        │           │   │E1││E2││..│  │  │
│        │           │   └──┘└──┘└──┘  │  │
│        │           └────────┬────────┘  │
│        └──── + residual ────┘           │
│                                         │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Final RMSNorm  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    LM Head      │   (tied with embedding weights)
└────────┬────────┘
         │
         ▼
     Logits
```

### Core Components

| Component | Description |
|---|---|
| **RMSNorm** | Root Mean Square normalization — simpler and faster than LayerNorm. Uses [Liger RMSNorm](https://github.com/linkedin/Liger-Kernel) when available, falls back to a manual PyTorch implementation. |
| **Rotary Position Embedding (RoPE)** | Encodes positional information by rotating query/key vectors. No learned position embeddings needed. |
| **Multi-Head Attention** | Standard causal (masked) self-attention. Supports [Flash Attention 2](https://github.com/Dao-AILab/flash-attention) for memory-efficient training, with PyTorch SDPA as fallback. |
| **SwiGLU Expert MLP** | Each expert uses a gated architecture: `SiLU(W_gate · x) ⊙ W_up · x`, projected back by `W_down`. This is the same expert design used by LLaMA and Mistral. |
| **Top-K Router** | A linear gate scores each token against all experts, selects the top-K, and uses softmax-normalized weights. Tokens are **shuffled** (grouped by expert) for batched computation, then **un-shuffled** back. |
| **Load Balancing Loss** | An auxiliary loss `α · N · Σ(mean_prob²)` that discourages the router from always picking the same experts, ensuring all experts get trained. |
| **Tied Embeddings** | The input embedding matrix and the output LM head share weights, reducing parameter count significantly. |

### GTLM-1-2B-A350M Configuration

| Parameter | Value |
|---|---|
| Hidden dimension | 1024 |
| Attention heads | 16 |
| Head dimension | 64 |
| Layers | 22 |
| Experts per layer | 16 |
| Top-K | 2 |
| Max sequence length | 2048 |
| Expert hidden multiplier | 5/3 |
| Vocabulary size | 32,002 |
| Tokenizer | `mistralai/Mistral-7B-v0.1` |

---

## Released Checkpoints

| Model | Parameters | Active | Description | Link |
|---|---|---|---|---|
| **GTLM-1-2B-A350M** | ~2B | ~350M | Main public checkpoint | [🤗 Hub](https://huggingface.co/Madras1/GTLM-1-2B-A350M) |
| **GTLM-1-2B-A350M-fp16** | ~2B | ~350M | FP16-oriented release | [🤗 Hub](https://huggingface.co/Madras1/GTLM-1-2B-A350M-fp16) |

### Training Setup

| Aspect | Details |
|---|---|
| Hardware | Single NVIDIA A100 |
| Duration | ~140 hours |
| Estimated cost | ~100 USD |
| Average throughput | ~50,000 tokens/second |
| Training tokens | ~15 billion |
| Precision | BF16 |
| Framework | PyTorch + DeepSpeed (ZeRO Stage 1) |
| Dataset | Curated English + Brazilian Portuguese blend |

### Dataset Composition

The training data emphasizes **dense reasoning signal** over raw scale:

| Category | Sources |
|---|---|
| Reasoning / Math | Nemotron Math, FineWeb-Edu, FineMath |
| General knowledge | Dolma, RedPajama, SlimPajama, FineWeb |
| Portuguese | Filtered Brazilian web documents |

---

## Benchmark Results

Zero-shot evaluation using [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-eval-harness):

| Model | Type | Active Params | Training Tokens | Avg | SciQ | ARC-E | PIQA | HellaSwag | OBQA | Winogrande |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TinyLlama 1.1B Chat | Dense | 1.1B | 3T | 63.5 | 88.3 | 61.8 | 74.5 | 60.4 | 35.4 | 60.3 |
| SmolLM 360M Inst | Dense | 360M | 600B | 60.7 | 85.9 | 64.1 | 70.6 | 52.8 | 37.0 | 53.7 |
| Qwen2 0.5B Inst | Dense | 0.5B | 12T | 59.0 | 90.3 | 55.0 | 69.3 | 49.1 | 33.0 | 57.3 |
| **GTLM-1.2B** | **MoE** | **350M** | **15B** | **56.2** | **87.6** | **56.6** | **66.7** | **42.0** | **32.6** | **51.8** |
| Pythia 410M | Dense | 410M | 300B | 53.9 | 80.8 | 51.9 | 67.3 | 40.6 | 29.6 | 53.4 |
| SmolLM 135M Inst | Dense | 135M | 600B | 52.9 | 73.4 | 49.2 | 67.3 | 42.0 | 33.8 | 51.4 |
| GPT-2 Medium | Dense | 355M | 10B | 52.5 | 77.1 | 49.3 | 66.2 | 39.5 | 30.0 | 53.0 |
| Pythia 160M | Dense | 160M | 300B | 47.7 | 73.4 | 43.4 | 61.4 | 30.4 | 27.4 | 50.0 |

> **Takeaway:** GTLM-1 outperforms GPT-2 Medium and Pythia 410M while using 15B tokens — orders of magnitude less data than most baselines. The **87.6% SciQ** score is particularly strong for a 350M-active-parameter model.

---

## Repository Structure

```text
GTLM/
├── PTLM.py                         # Backward-compatible DeepSpeed launcher
├── configuration_gtlm.py           # Hugging Face custom config (for inference)
├── modeling_gtlm.py                # Hugging Face custom model  (for inference)
├── pyproject.toml                   # Package metadata and dependencies
├── requirements.txt                 # Pip requirements
│
├── configs/
│   ├── train_base.json              # Default training configuration
│   └── gtlm_1_2b_a350m.example.json # Config matching the public checkpoint
│
├── src/gtlm/                        # Modular training package
│   ├── cli.py                       # CLI argument parsing and overrides
│   ├── config.py                    # Typed dataclass configuration
│   ├── data.py                      # Arrow dataset loading + DataLoader
│   ├── model.py                     # MoE model implementation (training)
│   ├── trainer.py                   # Training loop orchestration
│   ├── deepspeed_config.py          # DeepSpeed config builder
│   ├── checkpointing.py             # Checkpoint save/load helpers
│   └── monitoring.py                # Gradient norms, memory, numeric checks
│
├── scripts/
│   └── train.py                     # Explicit training entrypoint
│
├── tests/
│   └── test_config.py               # Configuration validation tests
│
├── docs/
│   ├── architecture.md              # Software architecture notes
│   ├── model-card-summary.md        # Public model card summary
│   └── training.md                  # Training pipeline documentation
│
└── legacy/
    └── PTLM_original.py             # Original monolithic training script
```

> **Design rationale:** The repository separates **training code** (`src/gtlm/`) from **Hugging Face inference code** (`modeling_gtlm.py` + `configuration_gtlm.py`). Training behavior is driven by JSON configs instead of hard-coded constants. The original script is preserved in `legacy/` for provenance. See [docs/architecture.md](docs/architecture.md) for the full design rationale.

---

## Getting Started

### Requirements

- **OS:** Linux with CUDA (required for training — Windows/macOS for code browsing only)
- **Python:** 3.10+
- **GPU:** NVIDIA GPU with CUDA support (A100 recommended; consumer GPUs work with smaller configs)
- **Core dependencies:** PyTorch ≥2.1, Transformers ≥4.40, DeepSpeed ≥0.14, Datasets ≥2.18

### Installation

```bash
# Clone the repository
git clone https://github.com/Madras1/GTLM.git
cd GTLM

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install as an editable package
pip install -e .
```

#### Optional CUDA Acceleration

These are not required — the code includes manual fallbacks — but they significantly improve training speed and memory efficiency:

```bash
pip install flash-attn    # Flash Attention 2 — faster, memory-efficient attention
pip install liger-kernel   # Liger RMSNorm — fused CUDA kernel for normalization
```

---

## Training

### Dataset Format

The training pipeline expects a directory of **Hugging Face Arrow shards**, each containing a column called `input_ids` with pre-tokenized integer sequences:

```text
/path/to/dataset/train/
├── shard_000.arrow
├── shard_001.arrow
├── shard_002.arrow
└── ...
```

Each row should look like:

```python
{"input_ids": [token_id_0, token_id_1, token_id_2, ...]}
```

**Important details:**
- The training loop applies **causal shifting** internally (`input = tokens[:-1]`, `labels = tokens[1:]`)
- Padding positions are masked with `-100` (ignored by CrossEntropyLoss)
- Rows should be **fixed-length** when using the default PyTorch collator
- Variable-length rows require a custom collator

### Launch Commands

**Single GPU — compatibility launcher:**
```bash
deepspeed --num_gpus=1 PTLM.py --config configs/train_base.json
```

**Single GPU — explicit script:**
```bash
deepspeed --num_gpus=1 scripts/train.py --config configs/train_base.json
```

**Using the public checkpoint config:**
```bash
deepspeed --num_gpus=1 scripts/train.py --config configs/gtlm_1_2b_a350m.example.json
```

**With CLI overrides:**
```bash
deepspeed --num_gpus=1 scripts/train.py \
  --config configs/train_base.json \
  --dataset_path /path/to/your/dataset/train \
  --output_dir /path/to/your/checkpoints \
  --tokenizer_name mistralai/Mistral-7B-v0.1
```

**With `torch.compile`:**
```bash
deepspeed --num_gpus=1 scripts/train.py \
  --config configs/train_base.json \
  --compile
```

### Configuration

All training hyperparameters are stored in JSON config files. The two provided configs are:

| Config | Purpose |
|---|---|
| [`configs/train_base.json`](configs/train_base.json) | Default small config for development and testing |
| [`configs/gtlm_1_2b_a350m.example.json`](configs/gtlm_1_2b_a350m.example.json) | Config matching the released GTLM-1 public checkpoint |

The configuration is organized into six sections:

| Section | Controls |
|---|---|
| `model` | Architecture shape (hidden dim, heads, layers, experts, top-k, etc.) |
| `data` | Tokenizer, dataset path, DataLoader settings |
| `optimizer` | Learning rate, betas, weight decay |
| `schedule` | Epochs, warmup ratio, min LR ratio |
| `runtime` | Output directory, checkpointing interval, logging, seeds |
| `deepspeed` | Micro batch size, gradient accumulation, ZeRO stage, BF16, offloading |

You can inspect and validate a config without starting training:

```bash
python PTLM.py --config configs/train_base.json --print_config --dry_run
```

### Resuming from Checkpoint

```bash
deepspeed --num_gpus=1 scripts/train.py \
  --config configs/train_base.json \
  --resume_checkpoint /path/to/checkpoint/directory
```

Checkpoints are saved as DeepSpeed checkpoint directories with the tag format `global_step{N}`. They store the model state, optimizer state, and client metadata (epoch, step, loss).

---

## Using the Model for Inference

The released checkpoints include Hugging Face custom-code files, so you can load and use the model directly:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Madras1/GTLM-1-2B-A350M"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)

input_text = "The meaning of life is"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

output = model.generate(input_ids, max_new_tokens=100, do_sample=True, temperature=0.7)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

> **Note:** You need `trust_remote_code=True` because GTLM uses custom `configuration_gtlm.py` and `modeling_gtlm.py` files hosted alongside the checkpoint.

---

## Engineering Checks

Run these lightweight checks to validate repository integrity without GPU or training:

```bash
# Validate config loading and schema
python PTLM.py --config configs/train_base.json --dry_run

# Run unit tests
python -m unittest discover -s tests

# Check syntax across all modules
python -m py_compile PTLM.py scripts/train.py src/gtlm/*.py
```

These checks do **not** train the model. They verify configuration loading, module wiring, and Python syntax.

---

## Documentation

Detailed documentation is available in the [`docs/`](docs/) directory:

| Document | Contents |
|---|---|
| [architecture.md](docs/architecture.md) | Software architecture, module responsibilities, and design rationale |
| [training.md](docs/training.md) | Training pipeline details, DeepSpeed config, dataset contract, preflight checklist |
| [model-card-summary.md](docs/model-card-summary.md) | Summary of the public Hugging Face model cards with benchmark tables |

---

## Reproducibility Notice

Exact reproduction of the released checkpoints requires more than just this config file. You would also need:

- The exact dataset shards and preprocessing
- The specific tokenizer revision
- Pinned dependency versions (PyTorch, DeepSpeed, Transformers, etc.)
- The same random seeds and hardware
- The checkpoint conversion process used for the Hugging Face release

This repository provides the **training recipe and architecture** — the released checkpoints are the **artifacts** of a specific training run.

---

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

<div align="center">

**Built with 🔥 by [Gabriel Yogi](https://huggingface.co/Madras1)**

*Proving that you don't need a datacenter to train a language model.*

</div>
