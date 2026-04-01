"""Tests for Grouped Query Attention."""

import torch
import pytest
from turboquant.model.config import ModelConfig
from turboquant.model.attention import GroupedQueryAttention
from turboquant.model.rope import precompute_freqs_cis


@pytest.fixture
def gqa_config():
    return ModelConfig(vocab_size=256, dim=64, n_layers=1, n_heads=4, n_kv_heads=2, max_seq_len=32)


def test_output_shape(gqa_config):
    attn = GroupedQueryAttention(gqa_config)
    B, T = 2, 8
    x = torch.randn(B, T, gqa_config.dim)
    freqs = precompute_freqs_cis(gqa_config.head_dim, T)
    out, cache = attn(x, freqs)
    assert out.shape == (B, T, gqa_config.dim)
    assert cache is None


def test_gradient_flows(gqa_config):
    attn = GroupedQueryAttention(gqa_config)
    B, T = 2, 4
    x = torch.randn(B, T, gqa_config.dim, requires_grad=True)
    freqs = precompute_freqs_cis(gqa_config.head_dim, T)
    out, _ = attn(x, freqs)
    out.sum().backward()
    assert x.grad is not None


def test_kv_cache_shape(gqa_config):
    attn = GroupedQueryAttention(gqa_config)
    B, T = 1, 4
    x = torch.randn(B, T, gqa_config.dim)
    freqs = precompute_freqs_cis(gqa_config.head_dim, gqa_config.max_seq_len)

    # Build KV cache
    k_cache = torch.zeros(B, gqa_config.max_seq_len, gqa_config.n_kv_heads, gqa_config.head_dim)
    v_cache = torch.zeros_like(k_cache)
    out, (k, v) = attn(x, freqs[:T], kv_cache=(k_cache, v_cache), cache_pos=0)
    assert k.shape == k_cache.shape
    assert v.shape == v_cache.shape


def test_causal_mask_applied(gqa_config):
    """Output at position i should not depend on positions j > i."""
    attn = GroupedQueryAttention(gqa_config)
    attn.eval()
    B, T = 1, 8
    x = torch.randn(B, T, gqa_config.dim)
    freqs = precompute_freqs_cis(gqa_config.head_dim, T)

    out1, _ = attn(x, freqs)
    # Modify future tokens
    x2 = x.clone()
    x2[:, 4:, :] += 10.0
    out2, _ = attn(x2, freqs)

    # Positions 0-3 should be unaffected
    assert torch.allclose(out1[:, :4, :], out2[:, :4, :], atol=1e-4)
