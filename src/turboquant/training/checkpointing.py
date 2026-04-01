"""Checkpoint save/load using safetensors.

safetensors is safe (no pickle), fast (zero-copy mmap), and ecosystem-standard.
Saves model weights + config JSON separately.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

from turboquant.model.config import ModelConfig


def save_checkpoint(
    model: nn.Module,
    config: ModelConfig,
    step: int,
    checkpoint_dir: str | Path,
    optimizer_state: dict | None = None,
) -> Path:
    """Save model weights as safetensors and config as JSON."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    step_dir = checkpoint_dir / f"step_{step:07d}"
    step_dir.mkdir(exist_ok=True)

    # Unwrap DDP/FSDP if needed
    raw_model = model.module if hasattr(model, "module") else model

    # Save weights (safetensors)
    state_dict = {k: v.cpu() for k, v in raw_model.state_dict().items()}
    save_file(state_dict, step_dir / "model.safetensors")

    # Save config
    with open(step_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    # Optionally save optimizer (for resuming training)
    if optimizer_state is not None:
        torch.save(optimizer_state, step_dir / "optimizer.pt")

    return step_dir


def load_checkpoint(
    checkpoint_dir: str | Path,
    model: nn.Module,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> tuple[nn.Module, ModelConfig]:
    """Load model weights and config from a checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)

    with open(checkpoint_dir / "config.json") as f:
        config = ModelConfig(**json.load(f))

    state_dict = load_file(checkpoint_dir / "model.safetensors", device=str(device))
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(state_dict, strict=strict)

    return model, config


def load_config(checkpoint_dir: str | Path) -> ModelConfig:
    with open(Path(checkpoint_dir) / "config.json") as f:
        return ModelConfig(**json.load(f))


def latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Return the latest step directory, or None if none exist."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    dirs = sorted(checkpoint_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    return dirs[-1] if dirs else None
