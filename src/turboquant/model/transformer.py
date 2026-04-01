"""Decoder-only Transformer.

Architecture: embedding → N × TransformerBlock → RMSNorm → LM head
Pre-norm residual: x = x + sublayer(norm(x))
No positional embedding in the embedding layer — RoPE handles it in attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from turboquant.model.attention import GroupedQueryAttention
from turboquant.model.config import ModelConfig
from turboquant.model.feedforward import SwiGLUFFN
from turboquant.model.normalization import RMSNorm
from turboquant.model.rope import precompute_freqs_cis


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn = GroupedQueryAttention(config)
        self.ffn = SwiGLUFFN(config)
        self.attn_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache=None,  # tuple[Tensor, Tensor] | CompressedKVCache | None
        cache_pos: int = 0,
    ):
        # Pre-norm attention with residual
        attn_out, updated_cache = self.attn(
            self.attn_norm(x), freqs_cis, mask, kv_cache, cache_pos
        )
        x = x + attn_out

        # Pre-norm FFN with residual
        x = x + self.ffn(self.ffn_norm(x))

        return x, updated_cache


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, eps=config.rms_norm_eps)

        # LM head — untied from embeddings (LLaMA-3 convention)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        # RoPE frequencies stored as a non-parameter buffer
        freqs = precompute_freqs_cis(
            config.head_dim, config.max_seq_len, theta=config.rope_theta
        )
        self.register_buffer("freqs_cis", freqs, persistent=False)

        self._use_grad_checkpoint = False

        self._init_weights()

    def _init_weights(self) -> None:
        """Small init, scaled by depth (GPT-2 / LLaMA convention)."""
        import math
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
        # Scale residual projections by 1/sqrt(2 * n_layers)
        scale = 1.0 / math.sqrt(2 * self.config.n_layers)
        for layer in self.layers:
            nn.init.normal_(layer.attn.wo.weight, mean=0.0, std=std * scale)
            nn.init.normal_(layer.ffn.w2.weight, mean=0.0, std=std * scale)

    def enable_gradient_checkpointing(self) -> None:
        self._use_grad_checkpoint = True

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_caches: list | None = None,  # list[tuple[Tensor,Tensor] | CompressedKVCache | None]
        cache_pos: int = 0,
    ) -> tuple[torch.Tensor, list]:
        """
        Args:
            input_ids: (batch, seq_len) long tensor
            mask: optional attention mask (for padding)
            kv_caches: per-layer KV caches for incremental decoding
            cache_pos: starting position in the KV cache

        Returns:
            logits: (batch, seq_len, vocab_size)
            updated_kv_caches: list of updated per-layer caches
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
        )

        x = self.embedding(input_ids)

        # Slice RoPE frequencies for current positions
        freqs_cis = self.freqs_cis[cache_pos : cache_pos + T]  # type: ignore[index]

        if kv_caches is None:
            kv_caches = [None] * self.config.n_layers

        updated_caches: list[tuple[torch.Tensor, torch.Tensor] | None] = []

        for i, layer in enumerate(self.layers):
            if self._use_grad_checkpoint and self.training:
                # Gradient checkpointing: recompute activations on backward pass
                def make_block_fn(blk: TransformerBlock):
                    def fn(x_in: torch.Tensor) -> torch.Tensor:
                        out, _ = blk(x_in, freqs_cis, mask, None, cache_pos)
                        return out
                    return fn
                x = grad_checkpoint(make_block_fn(layer), x, use_reentrant=False)
                updated_caches.append(None)
            else:
                x, updated_cache = layer(x, freqs_cis, mask, kv_caches[i], cache_pos)
                updated_caches.append(updated_cache)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits, updated_caches

    def n_params(self, exclude_embeddings: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            total -= self.embedding.weight.numel()
            if not self.config.tie_embeddings:
                total -= self.lm_head.weight.numel()
        return total
