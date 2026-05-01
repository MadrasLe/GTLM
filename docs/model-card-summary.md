# GTLM Model Card Summary

This page summarizes the public Hugging Face model cards for the GTLM-1 checkpoints. Source pages were checked on 2026-04-29.

## Source Models

| Model | Link | Notes |
| --- | --- | --- |
| GTLM-1-2B-A350M | https://huggingface.co/Madras1/GTLM-1-2B-A350M | Main GTLM-1 MoE release. |
| GTLM-1-2B-A350M-fp16 | https://huggingface.co/Madras1/GTLM-1-2B-A350M-fp16 | FP16-oriented release in the same model family. |

Both pages are tagged as text-generation, PyTorch, Safetensors, Mixture of Experts, custom code, English, and Apache-2.0.

## Reported Architecture

| Item | Reported value |
| --- | --- |
| Family | GTLM-1 |
| Model type | Sparse Mixture of Experts causal LM |
| Total parameter scale | 2B parameters |
| Active parameter scale | About 350M active parameters |
| Experts | 16 experts per layer |
| Routing | Top-2 |
| Dispatch | Vectorized token dispatch in PyTorch |
| Norm | Liger RMSNorm |
| Positional encoding | Manual RoPE |
| Expert MLP | Manual SwiGLU |

## Reported Training Setup

| Item | Reported value |
| --- | --- |
| Hardware | Single NVIDIA A100 |
| Duration | About 140 hours |
| Throughput | About 50,000 tokens/s average |
| Cost | About 100 USD |
| Stack | PyTorch + DeepSpeed |
| Training tokens | 15B |
| Language mix | Mostly English, with selected Brazilian Portuguese data |

## Dataset Blend

The card describes a curated mixture focused on dense reasoning signal rather than raw scale:

| Category | Examples named on the card |
| --- | --- |
| Reasoning/math | Nemotron Math, FineWeb-Edu, FineMath |
| General knowledge | Dolma, RedPajama, SlimPajama, FineWeb |
| Custom data | Filtered Brazilian web documents |

The Hugging Face metadata also lists dataset links including FineWeb, Tulu 3 SFT Mixture, and FineMath.

## Zero-Shot Evaluation

The card reports `lm-evaluation-harness` zero-shot results:

| Model | Type | Params active | Training tokens | Avg | SciQ | ARC-Easy | PIQA norm | HellaSwag norm | OpenBookQA norm | Winogrande |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TinyLlama 1.1B Chat | Dense | 1.1B | 3T | 63.5 | 88.3 | 61.8 | 74.5 | 60.4 | 35.4 | 60.3 |
| SmolLM 360M Inst | Dense | 360M | 600B | 60.7 | 85.9 | 64.1 | 70.6 | 52.8 | 37.0 | 53.7 |
| Qwen2 0.5B Inst | Dense | 0.5B | 12T | 59.0 | 90.3 | 55.0 | 69.3 | 49.1 | 33.0 | 57.3 |
| GTLM-1.2B | MoE | 350M | 15B | 56.2 | 87.6 | 56.6 | 66.7 | 42.0 | 32.6 | 51.8 |
| Pythia 410M | Dense | 410M | 300B | 53.9 | 80.8 | 51.9 | 67.3 | 40.6 | 29.6 | 53.4 |
| SmolLM 135M Inst | Dense | 135M | 600B | 52.9 | 73.4 | 49.2 | 67.3 | 42.0 | 33.8 | 51.4 |
| GPT-2 Medium | Dense | 355M | 10B | 52.5 | 77.1 | 49.3 | 66.2 | 39.5 | 30.0 | 53.0 |
| Pythia 160M | Dense | 160M | 300B | 47.7 | 73.4 | 43.4 | 61.4 | 30.4 | 27.4 | 50.0 |

## Interpretation

The strongest public claim is data efficiency: GTLM-1 is presented as a sparse MoE model that reaches competitive small-model benchmark scores after far fewer tokens than many dense baselines. The most striking individual number is SciQ, where the card reports 87.6 for GTLM-1.2B.

The careful interpretation is that this is an experimental research release. The model card frames GTLM-1 as an engineering and training-recipe demonstration, not as a state-of-the-art general assistant.
