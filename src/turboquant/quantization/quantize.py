"""Unified quantization interface.

quantize_model(model, method) is the single entry point for all quantization.
Walks the model and replaces nn.Linear layers with quantized equivalents.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from turboquant.quantization.int8 import Int8Linear
from turboquant.quantization.int4 import Int4Linear
from turboquant.quantization.dynamic import apply_dynamic_quantization


QuantMethod = Literal["int8", "int4", "dynamic"]


def quantize_model(
    model: nn.Module,
    method: QuantMethod = "int8",
    int4_group_size: int = 128,
    skip_modules: list[str] | None = None,
) -> nn.Module:
    """Replace all eligible nn.Linear layers with quantized equivalents.

    Args:
        model: The model to quantize (modified in-place for int8/int4).
        method: "int8" | "int4" | "dynamic"
        int4_group_size: Group size for INT4 quantization (default 128).
        skip_modules: List of module name patterns to skip (e.g., ["lm_head"]).

    Returns:
        The quantized model.
    """
    if method == "dynamic":
        return apply_dynamic_quantization(model)

    skip_modules = skip_modules or ["lm_head"]  # skip output projection by default

    _replace_linear_recursive(model, method, int4_group_size, skip_modules, parent_name="")
    return model


def _replace_linear_recursive(
    module: nn.Module,
    method: QuantMethod,
    group_size: int,
    skip_patterns: list[str],
    parent_name: str,
) -> None:
    for name, child in list(module.named_children()):
        full_name = f"{parent_name}.{name}" if parent_name else name

        if any(skip in full_name for skip in skip_patterns):
            continue

        if isinstance(child, nn.Linear):
            if method == "int8":
                new_layer = Int8Linear.from_float(child)
            elif method == "int4":
                # Only quantize if in_features is divisible by group_size
                if child.in_features % group_size == 0:
                    new_layer = Int4Linear.from_float(child, group_size=group_size)
                else:
                    continue
            else:
                continue
            setattr(module, name, new_layer)
        else:
            _replace_linear_recursive(child, method, group_size, skip_patterns, full_name)


def model_size_bytes(model: nn.Module) -> int:
    """Return approximate model size in bytes (parameters + buffers)."""
    total = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        total += tensor.nelement() * tensor.element_size()
    return total


def model_size_mb(model: nn.Module) -> float:
    return model_size_bytes(model) / (1024 ** 2)
