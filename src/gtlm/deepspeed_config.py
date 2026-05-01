"""DeepSpeed config builder."""

from __future__ import annotations

from .config import TrainingConfig


def build_deepspeed_config(config: TrainingConfig, world_size: int) -> dict:
    ds = config.deepspeed
    train_batch_size = (
        ds.train_micro_batch_size_per_gpu
        * ds.gradient_accumulation_steps
        * max(1, world_size)
    )

    output = {
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": ds.train_micro_batch_size_per_gpu,
        "gradient_accumulation_steps": ds.gradient_accumulation_steps,
        "steps_per_print": ds.steps_per_print,
        "bf16": {"enabled": ds.bf16_enabled},
        "zero_optimization": {
            "stage": ds.zero_stage,
            "offload_optimizer": {
                "device": ds.offload_optimizer_device,
                "pin_memory": ds.offload_pin_memory,
            },
            "overlap_comm": ds.overlap_comm,
            "contiguous_gradients": ds.contiguous_gradients,
            "reduce_scatter": ds.reduce_scatter,
            "reduce_bucket_size": "auto",
            "allgather_bucket_size": "auto",
            "round_robin_gradients": ds.round_robin_gradients,
        },
        "gradient_clipping": ds.gradient_clipping,
        "communication_data_type": "bf16" if ds.bf16_enabled else "fp16",
    }

    if ds.activation_checkpointing:
        output["activation_checkpointing"] = {
            "partition_activations": True,
            "cpu_checkpointing": ds.cpu_checkpointing,
            "contiguous_memory_optimization": True,
        }

    return output
