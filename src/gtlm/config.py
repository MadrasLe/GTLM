"""Configuration objects for GTLM training.

The project intentionally uses dataclasses plus JSON instead of a heavy config
framework. That keeps the training code easy to run in Colab, rented GPU
instances, and plain DeepSpeed launches.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int | None = None
    max_seq_length: int = 512
    hidden_dim: int = 512
    num_heads: int = 32
    num_experts: int = 4
    top_k: int = 2
    num_layers: int = 6
    dropout_rate: float = 0.05
    expert_hidden_multiplier: float = 5.0 / 3.0
    expert_hidden_multiple: int = 256
    init_std: float = 0.02
    prefer_liger_rmsnorm: bool = True
    prefer_flash_attention: bool = True

    def validate(self) -> None:
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if (self.hidden_dim // self.num_heads) % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.top_k < 1 or self.top_k > self.num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if self.max_seq_length < 2:
            raise ValueError("max_seq_length must be at least 2")


@dataclass
class DataConfig:
    tokenizer_name: str = "mistralai/Mistral-7B-v0.1"
    dataset_path: str = "/content/dataset/train"
    input_column: str = "input_ids"
    num_workers: int = 8
    pin_memory: bool = True
    drop_last: bool = True
    prefetch_factor: int = 4
    persistent_workers: bool = True
    shuffle: bool = True
    use_distributed_sampler: bool = True


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.01


@dataclass
class ScheduleConfig:
    num_epochs: int = 2
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.1


@dataclass
class RuntimeConfig:
    output_dir: str = "/content/drive/MyDrive/moe_checkpoints_liger_rmsnorm"
    checkpoint_every_steps: int = 500
    log_every_steps: int = 25
    load_balancing_alpha: float = 0.01
    compile_model: bool = False
    resume_checkpoint: str | None = None
    seed: int = 1337
    allow_tf32: bool = True
    empty_cache_before_train: bool = True


@dataclass
class DeepSpeedRuntimeConfig:
    train_micro_batch_size_per_gpu: int = 32
    gradient_accumulation_steps: int = 32
    steps_per_print: int = 25
    bf16_enabled: bool = True
    zero_stage: int = 1
    offload_optimizer_device: str = "cpu"
    offload_pin_memory: bool = True
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    reduce_scatter: bool = True
    round_robin_gradients: bool = True
    gradient_clipping: float = 1.0
    activation_checkpointing: bool = True
    cpu_checkpointing: bool = True


@dataclass
class TrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    deepspeed: DeepSpeedRuntimeConfig = field(default_factory=DeepSpeedRuntimeConfig)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainingConfig":
        return cls(
            model=ModelConfig(**values.get("model", {})),
            data=DataConfig(**values.get("data", {})),
            optimizer=OptimizerConfig(**values.get("optimizer", {})),
            schedule=ScheduleConfig(**values.get("schedule", {})),
            runtime=RuntimeConfig(**values.get("runtime", {})),
            deepspeed=DeepSpeedRuntimeConfig(**values.get("deepspeed", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        self.model.validate()
        if self.schedule.num_epochs < 1:
            raise ValueError("schedule.num_epochs must be at least 1")
        if self.deepspeed.train_micro_batch_size_per_gpu < 1:
            raise ValueError("train_micro_batch_size_per_gpu must be at least 1")
        if self.deepspeed.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        if self.runtime.log_every_steps < 1:
            raise ValueError("runtime.log_every_steps must be at least 1")


def load_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = TrainingConfig.from_dict(payload)
    config.validate()
    return config


def dump_config(config: TrainingConfig, path: str | Path) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
