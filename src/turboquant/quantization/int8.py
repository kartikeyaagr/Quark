"""INT8 weight-only quantization.

Per-channel symmetric quantization of linear layer weights.
Activations remain in float32/bfloat16; only weights are quantized to INT8.
This gives ~4x memory reduction vs float32 with minimal quality loss.

Quantization formula:
    scale = max(|W|) / 127        (per output channel)
    W_int8 = round(W / scale)
    W_dequant = W_int8 * scale    (applied at inference)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Int8Linear(nn.Module):
    """Drop-in replacement for nn.Linear with INT8 quantized weights.

    Weights are stored as int8 tensors. Dequantization happens at forward
    time before the matmul (weight-only quantization, activation stays fp).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # INT8 weights + float32 per-channel scales
        self.register_buffer("weight_int8", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(out_features, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features))
        else:
            self.bias = None

    @classmethod
    def from_float(cls, module: nn.Linear) -> "Int8Linear":
        """Quantize an existing nn.Linear to Int8Linear."""
        q = cls(module.in_features, module.out_features, bias=module.bias is not None)
        q.weight_int8, q.scale = quantize_weight_int8(module.weight.data)
        if module.bias is not None:
            q.bias = module.bias.data.clone()  # type: ignore[assignment]
        return q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize: (out_features, in_features) float
        weight = self.weight_int8.to(x.dtype) * self.scale.to(x.dtype).unsqueeze(1)
        return F.linear(x, weight, self.bias)  # type: ignore[arg-type]


def quantize_weight_int8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel symmetric quantization.

    Args:
        weight: (out_features, in_features) float tensor

    Returns:
        weight_int8: (out_features, in_features) int8 tensor
        scale: (out_features,) float32 per-channel scales
    """
    # Per output-channel max absolute value
    scale = weight.abs().max(dim=1).values / 127.0
    scale = scale.clamp(min=1e-8)

    weight_int8 = torch.round(weight / scale.unsqueeze(1)).clamp(-128, 127).to(torch.int8)
    return weight_int8, scale.to(torch.float32)
