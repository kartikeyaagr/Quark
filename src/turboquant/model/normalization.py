"""RMSNorm — Root Mean Square Layer Normalization.

Formula: x * rsqrt(mean(x^2) + eps) * weight
No bias, no mean subtraction. Faster and equivalent quality to LayerNorm
for pre-norm transformer architectures.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for numerical stability, then back
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * rms).to(x.dtype) * self.weight.to(x.dtype)
