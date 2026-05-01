# Repository Architecture

This repository is organized so the training project can grow without becoming
a single fragile script.

## Layers

```text
.
+-- PTLM.py                         # Backward-compatible DeepSpeed launcher
+-- configuration_gtlm.py            # Hugging Face custom config file
+-- modeling_gtlm.py                 # Hugging Face custom model file
+-- configs/                         # JSON experiment configs
+-- docs/                            # Human documentation
+-- legacy/PTLM_original.py          # Original single-file training script
+-- scripts/train.py                 # Explicit local training entrypoint
+-- src/gtlm/                        # Maintainable training package
+-- tests/                           # Lightweight engineering checks
```

## Responsibility Split

| Area | Path | Responsibility |
| --- | --- | --- |
| Compatibility launcher | `PTLM.py` | Keeps old `deepspeed PTLM.py` workflows working. |
| Training CLI | `src/gtlm/cli.py` | Parses command-line flags, loads config, applies overrides. |
| Config schema | `src/gtlm/config.py` | Dataclass-based experiment configuration and validation. |
| Model implementation | `src/gtlm/model.py` | MoE training model, attention, RoPE, experts, parameter counting. |
| Dataset loading | `src/gtlm/data.py` | Arrow shard loading and dataloader construction. |
| DeepSpeed config | `src/gtlm/deepspeed_config.py` | Builds DeepSpeed runtime dictionaries from typed config. |
| Training loop | `src/gtlm/trainer.py` | Tokenizer, optimizer, scheduler, DeepSpeed engine, loop, logging. |
| Checkpoints | `src/gtlm/checkpointing.py` | Save/load helper functions. |
| Monitoring | `src/gtlm/monitoring.py` | Numeric checks, gradient norm handling, CUDA memory helpers. |
| HF custom code | `configuration_gtlm.py`, `modeling_gtlm.py` | Files expected by Hugging Face custom model loading. |

## Why This Shape

- The root stays friendly for GitHub and Hugging Face readers.
- Training behavior is configured with JSON instead of hard-coded constants.
- The legacy script is preserved for provenance.
- The model, data, runtime config, and training loop are separately testable.
- Large artifacts are excluded from Git by `.gitignore`.
- The repo can later add evaluation, checkpoint conversion, or dataset preparation
  without stuffing more code into the launcher.
