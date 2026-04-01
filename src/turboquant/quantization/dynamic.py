"""Dynamic quantization using PyTorch's built-in torch.ao infrastructure.

Dynamic quantization computes activation scales at runtime (per-batch),
providing a good balance between quality and implementation simplicity.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """Apply PyTorch dynamic INT8 quantization to all Linear layers.

    This is the simplest form of quantization — no calibration dataset needed.
    Works well for CPU inference; less beneficial on GPU where BF16 is faster.
    """
    return torch.ao.quantization.quantize_dynamic(  # type: ignore[attr-defined]
        model,
        qconfig_spec={nn.Linear},
        dtype=torch.qint8,
    )
