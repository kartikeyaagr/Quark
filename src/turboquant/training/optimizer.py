"""AdamW optimizer configuration with proper weight decay groups.

1D parameters (norms, biases) are excluded from weight decay.
2D parameters (weight matrices) receive weight decay.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def configure_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    fused: bool = True,
) -> torch.optim.AdamW:
    """Build AdamW with correct weight decay parameter groups.

    Excludes 1D tensors (norms, biases) and embeddings from weight decay.
    Uses fused CUDA kernel when available for ~10% training speedup.
    """
    decay_params: list[torch.Tensor] = []
    no_decay_params: list[torch.Tensor] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() == 1 or "embedding" in name or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # Use fused AdamW on CUDA (available since PyTorch 2.0)
    use_fused = fused and torch.cuda.is_available()
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=betas,
        eps=eps,
        fused=use_fused,
    )
    return optimizer
