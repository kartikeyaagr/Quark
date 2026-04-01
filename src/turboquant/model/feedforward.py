"""SwiGLU Feed-Forward Network.

Three weight matrices: gate (w1), up (w3), down (w2).
Forward: w2(silu(w1(x)) * w3(x))

This gated architecture consistently outperforms vanilla FFN.
FFN hidden dim is set to (8/3 * dim) rounded to multiple of 256,
giving roughly the same parameter count as a 4x FFN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from turboquant.model.config import ModelConfig


class SwiGLUFFN(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.ffn_hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(config.ffn_hidden_dim, config.dim, bias=False)  # down
        self.w3 = nn.Linear(config.dim, config.ffn_hidden_dim, bias=False)  # up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: silu(gate) * up, then project down
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
