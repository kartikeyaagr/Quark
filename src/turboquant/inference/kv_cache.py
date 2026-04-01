"""Pre-allocated KV cache for efficient autoregressive generation.

Pre-allocating avoids repeated tensor creation/copying during the decode loop.
One KVCache instance per transformer layer.

Also re-exports build_compressed_kv_caches for convenience.
"""

from __future__ import annotations

import torch


class KVCache:
    """Fixed-size KV cache for a single transformer layer."""

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        shape = (batch_size, max_seq_len, n_kv_heads, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self._pos = 0

    @property
    def pos(self) -> int:
        return self._pos

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k, self.v

    def reset(self) -> None:
        self._pos = 0
        self.k.zero_()
        self.v.zero_()


from turboquant.inference.compressed_kv_cache import build_compressed_kv_caches  # noqa: F401,E402


def build_kv_caches(
    model_config: "ModelConfig",  # type: ignore[name-defined]  # avoid circular import
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Create one (k, v) cache pair per transformer layer."""
    from turboquant.model.config import ModelConfig
    caches = []
    shape = (batch_size, model_config.max_seq_len, model_config.n_kv_heads, model_config.head_dim)
    for _ in range(model_config.n_layers):
        k = torch.zeros(shape, device=device, dtype=dtype)
        v = torch.zeros(shape, device=device, dtype=dtype)
        caches.append((k, v))
    return caches
