"""Training metrics: loss tracking, perplexity, optional WandB logging."""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any


class MetricsTracker:
    """Tracks running loss and reports perplexity with optional WandB."""

    def __init__(
        self,
        window_size: int = 100,
        use_wandb: bool = False,
        wandb_project: str = "turboquant",
        wandb_config: dict[str, Any] | None = None,
    ) -> None:
        self.window_size = window_size
        self._losses: deque[float] = deque(maxlen=window_size)
        self._tokens: deque[int] = deque(maxlen=window_size)
        self._step = 0
        self._start_time = time.time()
        self._last_log_time = time.time()
        self._use_wandb = False

        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, config=wandb_config or {})
                self._use_wandb = True
            except ImportError:
                print("wandb not installed — skipping W&B logging")

    def update(self, loss: float, n_tokens: int) -> None:
        self._losses.append(loss)
        self._tokens.append(n_tokens)
        self._step += 1

    @property
    def smoothed_loss(self) -> float:
        if not self._losses:
            return float("inf")
        return sum(self._losses) / len(self._losses)

    @property
    def perplexity(self) -> float:
        try:
            return math.exp(self.smoothed_loss)
        except OverflowError:
            return float("inf")

    @property
    def tokens_per_second(self) -> float:
        elapsed = time.time() - self._start_time
        total_tokens = sum(self._tokens)
        return total_tokens / max(elapsed, 1e-9)

    def log(self, step: int, lr: float, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        metrics = {
            "step": step,
            "loss": self.smoothed_loss,
            "ppl": self.perplexity,
            "lr": lr,
            "tok/s": self.tokens_per_second,
        }
        if extra:
            metrics.update(extra)
        if self._use_wandb:
            import wandb
            wandb.log(metrics, step=step)
        return metrics

    def format_log(self, step: int, total_steps: int, lr: float) -> str:
        return (
            f"step {step:>7d}/{total_steps} | "
            f"loss {self.smoothed_loss:.4f} | "
            f"ppl {self.perplexity:.1f} | "
            f"lr {lr:.2e} | "
            f"{self.tokens_per_second:,.0f} tok/s"
        )
