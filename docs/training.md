# Training Notes

This document explains the modular training pipeline behind [../PTLM.py](../PTLM.py) and the main operational assumptions needed to run it safely.

## What PTLM.py Implements

[../PTLM.py](../PTLM.py) is now a compatibility launcher. The implementation lives in [../src/gtlm](../src/gtlm):

| Area | Implementation |
| --- | --- |
| Token embeddings | `src/gtlm/model.py`, `nn.Embedding` tied to the output `lm_head` |
| Attention | Multi-head causal attention with manual RoPE |
| Fast attention path | `flash_attn_func` when Flash Attention 2 is installed |
| Fallback attention path | PyTorch scaled dot-product attention |
| Normalization | `LigerRMSNorm` if installed, otherwise local `ManualRMSNorm` |
| MoE router | Linear gate plus Top-K expert selection |
| Experts | Manual SwiGLU MLP experts |
| Dispatch | Token shuffling grouped by expert, then `index_add_` to restore token order |
| Loss | Cross entropy plus auxiliary load-balancing loss |
| Optimizer | `src/gtlm/trainer.py`, `DeepSpeedCPUAdam` |
| Scheduler | `src/gtlm/trainer.py`, linear warmup followed by cosine decay |
| Checkpoints | `src/gtlm/checkpointing.py`, DeepSpeed checkpoint saves every configured interval |

## Key Constants To Review

The old script kept important training knobs near the top of the file. They now live in [../configs/train_base.json](../configs/train_base.json):

```python
MAX_SEQ_LENGTH = 512
HIDDEN_DIM = 512
NUM_HEADS = 32
NUM_EXPERTS = 4
TOP_K = 2
NUM_LAYERS = 6
DROPOUT_RATE = 0.05
LEARNING_RATE = 1e-3
LOAD_BALANCING_ALPHA = 0.01
NUM_EPOCHS = 2
CHECKPOINT_EVERY_STEPS = 500
```

The public model card describes the released GTLM-1 checkpoint as a 16-expert-per-layer Top-2 MoE. The matching example config is [../configs/gtlm_1_2b_a350m.example.json](../configs/gtlm_1_2b_a350m.example.json). If the goal is exact continuation or exact reproduction, also pin the dataset, tokenizer revision, dependency versions, seeds, and conversion path.

## Dataset Format

The script expects a directory like:

```text
/content/dataset/train/
+-- shard_000.arrow
+-- shard_001.arrow
+-- ...
```

Each row must contain:

```python
{"input_ids": [token_id_0, token_id_1, ...]}
```

Important details:

- The script does causal language-model shifting internally.
- `shift_input_ids` uses all tokens except the last.
- `shift_labels` uses all tokens except the first.
- Padding labels are replaced with `-100`, so PyTorch cross entropy ignores them.
- With the default PyTorch collator, `input_ids` should be fixed-length tensors. Variable-length rows need a custom collator.

## Path Assumptions

The default config uses Colab-style paths:

```python
DRIVE_PATH = "/content/drive/MyDrive/moe_checkpoints_liger_rmsnorm"
DATASET_PATH = "/content/dataset/train"
```

For a server run, change these in JSON or override them from the CLI before launching.

## Launch Commands

Single GPU through the compatibility launcher:

```bash
deepspeed --num_gpus=1 PTLM.py --config configs/train_base.json
```

Single GPU through the explicit script:

```bash
deepspeed --num_gpus=1 scripts/train.py --config configs/train_base.json
```

Single GPU with `torch.compile`:

```bash
deepspeed --num_gpus=1 scripts/train.py --config configs/train_base.json --compile
```

Resume:

```bash
deepspeed --num_gpus=1 scripts/train.py --config configs/train_base.json --resume_checkpoint /path/to/checkpoints
```

## DeepSpeed Configuration

The script builds its DeepSpeed config in Python:

```python
OPTIMIZED_DEEPSPEED_CONFIG = {
    "train_batch_size": 1024,
    "train_micro_batch_size_per_gpu": 32,
    "gradient_accumulation_steps": 32,
    "bf16": {"enabled": True},
    "zero_optimization": {"stage": 1, ...},
    "gradient_clipping": 1.0,
    "activation_checkpointing": {...},
}
```

At runtime, `train_batch_size` is recalculated as:

```python
micro_batch_size * gradient_accumulation_steps * world_size
```

That is useful, but it also means you should check the effective batch size whenever changing the GPU count.

## Checkpointing

Checkpoints are saved through:

```python
model_engine.save_checkpoint(save_path, tag=tag, client_state=client_state)
```

The tag format is:

```text
global_step{step}
```

The saved client state stores:

```python
{"epoch": epoch, "step": step, "loss": loss}
```

## Known Caveats

- Windows is not the intended runtime target for this script. Use Linux with CUDA for real training.
- `flash-attn` and `liger-kernel` are optional. The script falls back when they are missing.
- Multi-GPU data loading uses `DistributedSampler` by default, but you should still confirm shard balance and sample uniqueness for your dataset.
- `DeepSpeedCPUAdam` may require a compatible compiler/CUDA environment.
- Gradient norm handling accepts either tensors or Python floats.
- Exact reproducibility requires pinning dependency versions, tokenizer revision, dataset shards, seeds, and the final checkpoint conversion path.

## Suggested Preflight Checklist

Before a serious run:

1. Confirm the dataset folder contains `.arrow` files.
2. Confirm every row has fixed-length `input_ids`.
3. Confirm `tokenizer.pad_token` is valid.
4. Run a tiny dataset smoke test for 10 to 50 steps.
5. Watch loss, load-balancing loss, expert distribution, GPU memory, and gradient norm.
6. Save an early checkpoint and test resume before committing to a long run.
