"""Main training loop for TurboQuant.

Features:
- BF16 mixed precision (no GradScaler needed — BF16 has FP32 exponent range)
- Gradient clipping at 1.0
- Gradient accumulation (effective batch size = batch_size * grad_accum_steps)
- Gradient checkpointing (memory vs. speed tradeoff)
- Periodic checkpointing and logging
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from turboquant.model.config import ModelConfig
from turboquant.training.checkpointing import save_checkpoint, latest_checkpoint, load_checkpoint
from turboquant.training.metrics import MetricsTracker
from turboquant.training.optimizer import configure_optimizer
from turboquant.training.scheduler import get_cosine_schedule_with_warmup
from turboquant.training.distributed import is_main_process


@dataclass
class TrainingConfig:
    # Optimization
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    total_steps: int = 100_000
    min_lr_ratio: float = 0.1

    # Batch
    batch_size: int = 16
    grad_accum_steps: int = 1

    # Memory
    use_grad_checkpoint: bool = False
    compile: bool = False         # torch.compile for ~20% speedup

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 1000
    resume: bool = True

    # Logging
    log_every: int = 10
    use_wandb: bool = False
    wandb_project: str = "turboquant"


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        model_config: ModelConfig,
        train_config: TrainingConfig,
        train_loader: DataLoader,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model_config = model_config
        self.train_config = train_config
        self.device = torch.device(device)

        self.model = model.to(self.device)

        if train_config.use_grad_checkpoint:
            raw = self.model.module if hasattr(self.model, "module") else self.model
            if hasattr(raw, "enable_gradient_checkpointing"):
                raw.enable_gradient_checkpointing()

        if train_config.compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)  # type: ignore

        self.optimizer = configure_optimizer(
            self.model, lr=train_config.lr, weight_decay=train_config.weight_decay
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps=train_config.warmup_steps,
            total_steps=train_config.total_steps,
            min_lr_ratio=train_config.min_lr_ratio,
        )
        self.train_loader = train_loader
        self.metrics = MetricsTracker(
            use_wandb=train_config.use_wandb,
            wandb_project=train_config.wandb_project,
            wandb_config={**model_config.to_dict(), **train_config.__dict__},
        )

        self.step = 0
        self._resume_if_needed()

    def _resume_if_needed(self) -> None:
        if not self.train_config.resume:
            return
        ckpt = latest_checkpoint(self.train_config.checkpoint_dir)
        if ckpt is None:
            return
        # Extract step from directory name
        self.step = int(ckpt.name.split("_")[1])
        raw = self.model.module if hasattr(self.model, "module") else self.model
        load_checkpoint(ckpt, raw, device=self.device)
        if is_main_process():
            print(f"Resumed from checkpoint: {ckpt} (step {self.step})")

    def _compute_loss(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        logits, _ = self.model(input_ids)
        # Flatten for cross-entropy: (B*T, vocab) vs (B*T,)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

    def train(self) -> None:
        tc = self.train_config
        self.model.train()

        # BF16 autocast context (no GradScaler needed for BF16)
        use_bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        autocast_ctx = torch.autocast(device_type=self.device.type, dtype=dtype)

        loader_iter = iter(self.train_loader)

        while self.step < tc.total_steps:
            self.optimizer.zero_grad()
            total_loss = 0.0
            total_tokens = 0

            for accum_step in range(tc.grad_accum_steps):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(self.train_loader)
                    batch = next(loader_iter)

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                with autocast_ctx:
                    loss = self._compute_loss(input_ids, labels)
                    loss = loss / tc.grad_accum_steps

                loss.backward()
                total_loss += loss.item()
                total_tokens += input_ids.numel()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)

            self.optimizer.step()
            self.scheduler.step()
            self.step += 1

            # Metrics
            current_lr = self.scheduler.get_last_lr()[0]
            self.metrics.update(total_loss, total_tokens)

            if is_main_process() and self.step % tc.log_every == 0:
                print(self.metrics.format_log(self.step, tc.total_steps, current_lr))

            if is_main_process() and self.step % tc.checkpoint_every == 0:
                raw = self.model.module if hasattr(self.model, "module") else self.model
                save_checkpoint(raw, self.model_config, self.step, tc.checkpoint_dir)

        if is_main_process():
            raw = self.model.module if hasattr(self.model, "module") else self.model
            save_checkpoint(raw, self.model_config, self.step, tc.checkpoint_dir)
            print(f"Training complete. Final checkpoint at step {self.step}.")
