"""DeepSpeed training orchestration."""

from __future__ import annotations

from argparse import Namespace
import os
import random
import time

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from transformers import AutoTokenizer

from .checkpointing import load_checkpoint, save_checkpoint
from .config import TrainingConfig
from .data import ArrowTokenDataset, build_train_dataloader
from .deepspeed_config import build_deepspeed_config
from .model import (
    FLASH_ATTN2_AVAILABLE,
    LIGER_RMSNORM_AVAILABLE,
    GTLMTrainingModel,
    count_parameters,
)
from .monitoring import cuda_memory_gb, get_grad_norm, is_invalid_number, max_cuda_memory_gb


def configure_torch_runtime(config: TrainingConfig) -> None:
    torch.manual_seed(config.runtime.seed)
    random.seed(config.runtime.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.runtime.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = config.runtime.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.runtime.allow_tf32
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)


def print_model_info(model: GTLMTrainingModel) -> None:
    stats = count_parameters(model)
    print("\n" + "=" * 80)
    print("GTLM MoE training model")
    print("=" * 80)
    print(
        "Architecture: "
        f"{model.config.num_layers} layers | "
        f"{model.config.num_experts} experts | "
        f"Top-{model.config.top_k}"
    )
    print(f"Total parameters:  {stats['total_params'] / 1e9:.3f}B ({stats['total_params']:,})")
    print(f"Active parameters: {stats['active_params'] / 1e9:.3f}B ({stats['active_params']:,})")
    print(f"Active ratio:      {stats['utilization_ratio']:.2f}% per token")
    print("-" * 80)
    print(f"Liger RMSNorm:     {'available' if LIGER_RMSNORM_AVAILABLE else 'not available'}")
    print(f"Flash Attention 2: {'available' if FLASH_ATTN2_AVAILABLE else 'not available'}")
    print("=" * 80)


def _create_optimizer_and_scheduler(
    model: GTLMTrainingModel,
    config: TrainingConfig,
    steps_per_epoch: int,
):
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    opt_cfg = config.optimizer
    optimizer = DeepSpeedCPUAdam(
        model.parameters(),
        lr=opt_cfg.learning_rate,
        betas=(opt_cfg.beta1, opt_cfg.beta2),
        eps=opt_cfg.eps,
        weight_decay=opt_cfg.weight_decay,
    )

    total_global_steps = max(1, steps_per_epoch * config.schedule.num_epochs)
    warmup_steps = max(1, int(config.schedule.warmup_ratio * total_global_steps))
    cosine_steps = max(1, total_global_steps - warmup_steps)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=opt_cfg.learning_rate * config.schedule.min_lr_ratio,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    return optimizer, scheduler


def train(config: TrainingConfig, args: Namespace) -> None:
    import deepspeed

    config.validate()
    configure_torch_runtime(config)

    os.makedirs(config.runtime.output_dir, exist_ok=True)

    print(f"Loading tokenizer: {config.data.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset from: {config.data.dataset_path}")
    train_dataset = ArrowTokenDataset(
        config.data.dataset_path,
        input_column=config.data.input_column,
    )

    vocab_size = config.model.vocab_size or tokenizer.vocab_size
    model = GTLMTrainingModel(config.model, vocab_size=vocab_size)
    print_model_info(model)

    if config.runtime.compile_model:
        print("Applying torch.compile(mode='max-autotune')")
        model = torch.compile(model, mode="max-autotune")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    deepspeed_config = build_deepspeed_config(config, world_size)

    train_loader = build_train_dataloader(
        train_dataset,
        config.data,
        batch_size=config.deepspeed.train_micro_batch_size_per_gpu,
        world_size=world_size,
        rank=rank,
    )

    steps_per_epoch = max(
        1,
        len(train_loader) // config.deepspeed.gradient_accumulation_steps,
    )
    optimizer, scheduler = _create_optimizer_and_scheduler(model, config, steps_per_epoch)

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=deepspeed_config,
    )

    resume_checkpoint = config.runtime.resume_checkpoint
    start_epoch, global_step, _ = 0, 0, float("inf")
    if resume_checkpoint:
        start_epoch, global_step, _ = load_checkpoint(model_engine, resume_checkpoint)

    model_engine.train()
    start_time = time.time()
    total_tokens = 0
    recent_losses: list[float] = []

    if torch.cuda.is_available() and config.runtime.empty_cache_before_train:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    print("\nStarting GTLM MoE training")

    for epoch in range(start_epoch, config.schedule.num_epochs):
        sampler = getattr(train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.schedule.num_epochs}",
            disable=(getattr(model_engine, "local_rank", 0) != 0),
        )

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(model_engine.device, non_blocking=True).long()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_input_ids = input_ids[:, :-1].contiguous()
            pad_mask = shift_labels != tokenizer.pad_token_id
            shift_labels_masked = shift_labels.masked_fill(~pad_mask, -100)

            lm_loss, lb_loss, _ = model_engine(shift_input_ids, shift_labels_masked)

            if lm_loss is None or is_invalid_number(lm_loss):
                if model_engine.local_rank == 0:
                    print(f"\nSkipping step {global_step}: invalid loss={lm_loss}\n")
                model_engine.zero_grad()
                continue

            loss = lm_loss + config.runtime.load_balancing_alpha * lb_loss
            model_engine.backward(loss)
            grad_norm = get_grad_norm(model_engine)
            model_engine.step()

            total_tokens += pad_mask.sum().item()
            recent_losses.append(float(lm_loss.detach().float().cpu().item()))

            if model_engine.is_gradient_accumulation_boundary():
                global_step += 1

                should_log = (
                    model_engine.local_rank == 0
                    and global_step % config.runtime.log_every_steps == 0
                )
                if should_log:
                    avg_loss = sum(recent_losses[-10:]) / min(10, len(recent_losses))
                    elapsed = time.time() - start_time
                    tps = total_tokens / elapsed if elapsed > 0 else 0.0
                    current_lr = optimizer.param_groups[0]["lr"]
                    progress_bar.set_postfix(
                        {
                            "loss": f"{avg_loss:.3f}",
                            "lr": f"{current_lr:.2e}",
                            "grad": f"{grad_norm:.3f}",
                            "tps": f"{tps:,.0f}",
                            "gpu": f"{cuda_memory_gb():.1f}GB",
                        }
                    )

                    if is_invalid_number(grad_norm):
                        print("\nWarning: gradient norm is NaN/inf")
                    elif grad_norm < 1e-6 and global_step > 100:
                        print("\nWarning: gradient norm is very small")

                should_checkpoint = (
                    global_step > 0
                    and global_step % config.runtime.checkpoint_every_steps == 0
                    and model_engine.local_rank == 0
                )
                if should_checkpoint:
                    current_loss = sum(recent_losses[-20:]) / min(20, len(recent_losses))
                    save_checkpoint(
                        model_engine,
                        epoch,
                        global_step,
                        current_loss,
                        config.runtime.output_dir,
                    )

        if model_engine.local_rank == 0:
            window = recent_losses[-max(1, len(train_loader)) :]
            current_loss = sum(window) / len(window) if window else 0.0
            print(f"\nEpoch {epoch + 1} complete - loss={current_loss:.4f}")
            save_checkpoint(
                model_engine,
                epoch,
                global_step,
                current_loss,
                config.runtime.output_dir,
            )

    total_time = time.time() - start_time
    if model_engine.local_rank == 0:
        final_tps = total_tokens / total_time if total_time > 0 else 0.0
        print("\n" + "=" * 70)
        print("Training complete")
        print("=" * 70)
        print(f"Time:       {total_time / 3600:.2f} hours")
        print(f"Final TPS:  {final_tps:,.0f}")
        print(f"GPU memory: {max_cuda_memory_gb():.2f}GB")
        print("=" * 70)
