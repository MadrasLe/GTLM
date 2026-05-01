"""Checkpoint helpers."""

from __future__ import annotations

import os


def save_checkpoint(model_engine, epoch: int, step: int, loss: float, save_path: str) -> None:
    tag = f"global_step{step}"
    client_state = {"epoch": epoch, "step": step, "loss": loss}
    model_engine.save_checkpoint(save_path, tag=tag, client_state=client_state)
    print(f"Checkpoint saved: {os.path.join(save_path, tag)}")


def load_checkpoint(model_engine, checkpoint_path: str) -> tuple[int, int, float]:
    load_path, client_state = model_engine.load_checkpoint(checkpoint_path)
    if load_path:
        epoch = client_state.get("epoch", 0)
        step = client_state.get("step", 0)
        loss = client_state.get("loss", float("inf"))
        print(f"Checkpoint loaded: {load_path}\n   -> epoch={epoch}, step={step}")
        return epoch, step, loss

    print("No checkpoint found.")
    return 0, 0, float("inf")
