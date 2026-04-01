"""Distributed training helpers for DDP and FSDP.

Usage:
    # DDP (single-node multi-GPU, model fits in one GPU)
    setup_ddp(rank, world_size)
    model = wrap_ddp(model, device)

    # FSDP (model too large for single GPU)
    setup_ddp(rank, world_size)  # same init_process_group
    model = wrap_fsdp(model)
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp(rank: int, world_size: int, backend: str = "nccl") -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_ddp(model: nn.Module, device: torch.device) -> DDP:
    model = model.to(device)
    return DDP(model, device_ids=[device.index])


def wrap_fsdp(model: nn.Module) -> nn.Module:
    """Wrap model with FSDP for large model training."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        import functools
        from turboquant.model.transformer import TransformerBlock

        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerBlock},
        )
        return FSDP(model, auto_wrap_policy=wrap_policy)
    except ImportError:
        raise RuntimeError("FSDP requires PyTorch >= 2.0 with CUDA support")


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1
