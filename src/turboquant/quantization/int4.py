"""INT4 group-wise weight quantization.

Group-wise quantization (group_size=128) preserves quality vs per-tensor INT4.
Two INT4 values are packed into a single uint8 byte for 2x storage efficiency.

This follows the GPTQ/AWQ-style approach:
    - Divide weight rows into groups of `group_size` elements
    - Compute scale and zero_point per group
    - Quantize: q = round((w - zero_point) / scale)  ∈ [0, 15]
    - Dequantize: w ≈ q * scale + zero_point
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Int4Linear(nn.Module):
    """Drop-in replacement for nn.Linear with INT4 group-wise quantized weights.

    Weights stored as packed uint8 (2 INT4 values per byte).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert in_features % group_size == 0, (
            f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        n_groups = in_features // group_size

        # Packed INT4 weights: each pair of INT4 values packed into one uint8
        # Shape: (out_features, in_features // 2) uint8
        self.register_buffer(
            "weight_packed",
            torch.zeros(out_features, in_features // 2, dtype=torch.uint8),
        )
        # Per-group scales and zero points: (out_features, n_groups)
        self.register_buffer("scale", torch.ones(out_features, n_groups, dtype=torch.float32))
        self.register_buffer("zero_point", torch.zeros(out_features, n_groups, dtype=torch.float32))

        if bias:
            self.register_buffer("bias", torch.zeros(out_features))
        else:
            self.bias = None

    @classmethod
    def from_float(cls, module: nn.Linear, group_size: int = 128) -> "Int4Linear":
        """Quantize an existing nn.Linear to Int4Linear."""
        in_f = module.in_features
        out_f = module.out_features
        q = cls(in_f, out_f, group_size=group_size, bias=module.bias is not None)

        weight_int4, scale, zero_point = quantize_weight_int4(
            module.weight.data, group_size=group_size
        )
        q.weight_packed = pack_int4(weight_int4)
        q.scale = scale
        q.zero_point = zero_point

        if module.bias is not None:
            q.bias = module.bias.data.clone()  # type: ignore[assignment]
        return q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = dequantize_weight_int4(
            unpack_int4(self.weight_packed, self.in_features),
            self.scale,
            self.zero_point,
            self.group_size,
        ).to(x.dtype)
        return F.linear(x, weight, self.bias)  # type: ignore[arg-type]


def quantize_weight_int4(
    weight: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group-wise asymmetric INT4 quantization.

    Returns:
        weight_int4: (out_features, in_features) uint8 [0-15]
        scale: (out_features, n_groups) float32
        zero_point: (out_features, n_groups) float32
    """
    out_f, in_f = weight.shape
    n_groups = in_f // group_size

    # Reshape to (out_f, n_groups, group_size)
    w = weight.view(out_f, n_groups, group_size)

    w_min = w.min(dim=-1).values          # (out_f, n_groups)
    w_max = w.max(dim=-1).values

    scale = (w_max - w_min) / 15.0        # range [0, 15] = 4 bits
    scale = scale.clamp(min=1e-8)
    zero_point = w_min                    # asymmetric: zero maps to w_min

    w_norm = (w - zero_point.unsqueeze(-1)) / scale.unsqueeze(-1)
    weight_int4 = w_norm.round().clamp(0, 15).to(torch.uint8)
    weight_int4 = weight_int4.view(out_f, in_f)

    return weight_int4, scale, zero_point


def pack_int4(weight_int4: torch.Tensor) -> torch.Tensor:
    """Pack pairs of INT4 values into uint8 bytes.

    Input:  (out_features, in_features)  uint8, values in [0, 15]
    Output: (out_features, in_features // 2)  uint8
    """
    assert weight_int4.shape[1] % 2 == 0
    high = weight_int4[:, 0::2]  # even columns → high nibble
    low = weight_int4[:, 1::2]   # odd columns → low nibble
    return (high << 4) | low


def unpack_int4(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """Unpack uint8 bytes back to individual INT4 values.

    Input:  (out_features, in_features // 2)  uint8
    Output: (out_features, in_features)  uint8, values in [0, 15]
    """
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    result = torch.empty(packed.shape[0], in_features, dtype=torch.uint8, device=packed.device)
    result[:, 0::2] = high
    result[:, 1::2] = low
    return result


def dequantize_weight_int4(
    weight_int4: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Dequantize INT4 weights back to float."""
    out_f, in_f = weight_int4.shape
    n_groups = in_f // group_size

    w = weight_int4.float().view(out_f, n_groups, group_size)
    w = w * scale.unsqueeze(-1) + zero_point.unsqueeze(-1)
    return w.view(out_f, in_f)
