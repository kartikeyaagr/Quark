"""Compressed KV cache using TurboQuant vector quantization.

Drop-in replacement for the plain (k, v) tuple used by attention.py.
On each write the keys/values are quantized; on each read they are
dequantized back to full precision before being passed to SDPA.

Memory layout (per token per head, at 3-bit effective precision):
  K (or V):
    mse_indices:    uint8   — dim bytes   (codebook index per coordinate, ≤4 bits → fits uint8)
    qjl_packed:     uint8   — dim//8 bytes (8 sign bits per byte)
    residual_norm:  float16 — 2 bytes     (‖r‖₂ scalar)
    mse_scale:      float16 — 2 bytes     (rotation scale scalar)
  Total K: dim * (1 + 1/8) + 4  bytes  vs  dim * 4  bytes for FP32
  For dim=64:  76 bytes compressed vs 256 bytes FP32 → 3.4× reduction

Usage:
    cache = CompressedKVCache(batch_size, max_seq_len, n_kv_heads, head_dim,
                               device, bits=3)
    cache.write(k, v, cache_pos)
    k_full, v_full = cache.read(length=cache_pos + T, batch_size=B)
"""

from __future__ import annotations

import torch

from turboquant.quantization.turboquant_vq import TurboQuantProd, TurboQuantProdPackage


class CompressedKVCache:
    """TurboQuant-compressed KV cache for a single transformer layer.

    Args:
        batch_size: maximum batch size
        max_seq_len: maximum number of tokens to cache
        n_kv_heads: number of KV attention heads
        head_dim: dimension per head (must be divisible by 8)
        device: torch device
        bits: effective bit-width per coordinate (2–4)
        dtype: dtype for dequantized outputs (default float32)
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device,
        bits: int = 3,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if head_dim % 8 != 0:
            raise ValueError(f"head_dim must be divisible by 8, got {head_dim}")
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.bits = bits
        self.dtype = dtype

        self._tq = TurboQuantProd(dim=head_dim, bits=bits)

        # Storage shapes
        idx_shape    = (batch_size, max_seq_len, n_kv_heads, head_dim)
        packed_shape = (batch_size, max_seq_len, n_kv_heads, head_dim // 8)
        scalar_shape = (batch_size, max_seq_len, n_kv_heads)

        # Keys
        self._k_mse_idx   = torch.zeros(idx_shape,    dtype=torch.uint8,   device=device)
        self._k_qjl       = torch.zeros(packed_shape,  dtype=torch.uint8,   device=device)
        self._k_res_norm  = torch.zeros(scalar_shape,  dtype=torch.float16, device=device)
        self._k_mse_scale = torch.zeros(scalar_shape,  dtype=torch.float16, device=device)

        # Values
        self._v_mse_idx   = torch.zeros(idx_shape,    dtype=torch.uint8,   device=device)
        self._v_qjl       = torch.zeros(packed_shape,  dtype=torch.uint8,   device=device)
        self._v_res_norm  = torch.zeros(scalar_shape,  dtype=torch.float16, device=device)
        self._v_mse_scale = torch.zeros(scalar_shape,  dtype=torch.float16, device=device)

    def write(self, k: torch.Tensor, v: torch.Tensor, cache_pos: int) -> None:
        """Quantize and store K and V for positions [cache_pos, cache_pos + T).

        Args:
            k: (B, T, n_kv_heads, head_dim)
            v: (B, T, n_kv_heads, head_dim)
            cache_pos: starting position in the sequence
        """
        B, T = k.shape[0], k.shape[1]

        def _write_one(x, mse_idx, qjl, res_norm, mse_scale):
            pkg = self._tq.quantize(x.float())
            mse_idx[:B, cache_pos : cache_pos + T]   = pkg.mse_indices[:B]
            qjl[:B, cache_pos : cache_pos + T]        = pkg.qjl_packed[:B]
            res_norm[:B, cache_pos : cache_pos + T]   = pkg.residual_norms[:B].to(torch.float16)
            mse_scale[:B, cache_pos : cache_pos + T]  = pkg.mse_scales[:B].to(torch.float16)

        _write_one(k, self._k_mse_idx, self._k_qjl, self._k_res_norm, self._k_mse_scale)
        _write_one(v, self._v_mse_idx, self._v_qjl, self._v_res_norm, self._v_mse_scale)

    def read(
        self, length: int, batch_size: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize and return K and V for positions [0, length).

        Args:
            length: number of positions to read
            batch_size: effective batch size (defaults to self.batch_size)

        Returns:
            k: (B, length, n_kv_heads, head_dim) full precision
            v: (B, length, n_kv_heads, head_dim) full precision
        """
        B = batch_size if batch_size is not None else self.batch_size

        def _read_one(mse_idx, qjl, res_norm, mse_scale):
            pkg = TurboQuantProdPackage(
                mse_indices=mse_idx[:B, :length],
                qjl_packed=qjl[:B, :length],
                residual_norms=res_norm[:B, :length].float(),
                mse_scales=mse_scale[:B, :length].float(),
            )
            return self._tq.dequantize(pkg).to(self.dtype)

        k = _read_one(self._k_mse_idx, self._k_qjl, self._k_res_norm, self._k_mse_scale)
        v = _read_one(self._v_mse_idx, self._v_qjl, self._v_res_norm, self._v_mse_scale)
        return k, v

    def reset(self) -> None:
        """Clear all cached values."""
        for buf in (
            self._k_mse_idx, self._k_qjl, self._k_res_norm, self._k_mse_scale,
            self._v_mse_idx, self._v_qjl, self._v_res_norm, self._v_mse_scale,
        ):
            buf.zero_()


def build_compressed_kv_caches(
    model_config: "ModelConfig",  # type: ignore[name-defined]
    batch_size: int,
    device: torch.device,
    bits: int = 3,
    dtype: torch.dtype = torch.float32,
) -> list[CompressedKVCache]:
    """Create one CompressedKVCache per transformer layer."""
    from turboquant.model.config import ModelConfig
    return [
        CompressedKVCache(
            batch_size=batch_size,
            max_seq_len=model_config.max_seq_len,
            n_kv_heads=model_config.n_kv_heads,
            head_dim=model_config.head_dim,
            device=device,
            bits=bits,
            dtype=dtype,
        )
        for _ in range(model_config.n_layers)
    ]
