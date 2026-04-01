"""Grouped Query Attention (GQA) with optional Exclusive Self Attention (XSA).

Uses F.scaled_dot_product_attention which automatically dispatches to
FlashAttention-2/3 on supported hardware (CUDA with appropriate GPU).
KV heads are fewer than query heads (GQA), saving memory during inference.
No bias on any projection. No dropout in projections (handled at block level).

XSA (Apple, arxiv 2603.09078): after standard SDPA, subtract the component of
the output aligned with the token's own value vector, forcing attention to
encode only contextual information orthogonal to the self-representation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from turboquant.model.config import ModelConfig
from turboquant.model.rope import apply_rotary_emb


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_query_groups = config.n_query_groups
        self.dropout = config.dropout
        self.use_xsa = config.use_xsa

        # Projections — no bias (standard modern practice)
        self.wq = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: "tuple[torch.Tensor, torch.Tensor] | CompressedKVCache | None" = None,
        cache_pos: int = 0,
    ) -> "tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | CompressedKVCache | None]":
        """
        Args:
            x: (batch, seq_len, dim)
            freqs_cis: (seq_len, head_dim // 2) complex
            mask: optional attention mask
            kv_cache: (k_cache, v_cache) tensors for autoregressive generation
            cache_pos: current position in the cache (for incremental decoding)

        Returns:
            output: (batch, seq_len, dim)
            updated kv_cache (or None if not using cache)
        """
        B, T, _ = x.shape

        # Project to Q, K, V
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # Apply RoPE to Q and K
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # Save current-token V before cache merge for XSA.
        # Shape: (B, T_new, n_kv_heads, head_dim) — the self values for the new tokens only.
        v_new = v

        # Update KV cache if provided
        # Supports both plain tensor caches and CompressedKVCache objects.
        if kv_cache is not None:
            from turboquant.inference.compressed_kv_cache import CompressedKVCache
            if isinstance(kv_cache, CompressedKVCache):
                kv_cache.write(k, v, cache_pos)
                k, v = kv_cache.read(length=cache_pos + T, batch_size=B)
                updated_cache = kv_cache
            else:
                k_cache, v_cache = kv_cache
                k_cache[:B, cache_pos : cache_pos + T] = k
                v_cache[:B, cache_pos : cache_pos + T] = v
                k = k_cache[:B, : cache_pos + T]
                v = v_cache[:B, : cache_pos + T]
                updated_cache = (k_cache, v_cache)
        else:
            updated_cache = None

        # GQA: expand KV heads to match query heads
        # (B, seq, n_kv_heads, head_dim) → (B, seq, n_heads, head_dim)
        if self.n_query_groups > 1:
            k = k.repeat_interleave(self.n_query_groups, dim=2)
            v = v.repeat_interleave(self.n_query_groups, dim=2)

        # Transpose to (B, heads, seq, head_dim) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Causal masking strategy:
        # - No cache (prefill only): is_causal=True works correctly.
        # - With cache, Q_len == KV_len (first prefill with cache): is_causal=True is correct.
        # - With cache, Q_len < KV_len (incremental decode): is_causal=True applies a
        #   lower-triangular mask of shape (Q_len, KV_len) which is WRONG — it would block
        #   the query from attending to earlier cached tokens. Use is_causal=False instead
        #   since the cache already ensures causality by construction.
        kv_len = k.shape[2]
        use_causal_flag = mask is None and (T == kv_len)
        if mask is None and T < kv_len:
            # Decode step: allow attending to all cached tokens (all 1s = no mask needed)
            explicit_mask = None
            is_causal = False
        else:
            explicit_mask = mask
            is_causal = use_causal_flag

        # Flash Attention via PyTorch's fused SDPA kernel
        attn_dropout = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=explicit_mask,
            dropout_p=attn_dropout,
            is_causal=is_causal,
        )

        # XSA: subtract self-value projection from attention output.
        # This forces attention output to be orthogonal to the token's own value vector,
        # eliminating "attention similarity bias" (Apple XSA paper, arxiv 2603.09078).
        if self.use_xsa:
            # v_new: (B, T_new, n_kv_heads, head_dim) → expand GQA → transpose
            v_self = v_new
            if self.n_query_groups > 1:
                v_self = v_self.repeat_interleave(self.n_query_groups, dim=2)
            v_self = v_self.transpose(1, 2)  # (B, n_heads, T_new, head_dim)
            # Normalize self-value vectors along head_dim
            v_self_n = F.normalize(v_self, dim=-1)
            # Subtract the component of out aligned with v_self
            out = out - (out * v_self_n).sum(dim=-1, keepdim=True) * v_self_n

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out), updated_cache
