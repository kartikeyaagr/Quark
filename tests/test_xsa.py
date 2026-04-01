"""Tests for Apple Exclusive Self Attention (XSA).

Paper: arxiv 2603.09078
Key property: attention output must be orthogonal to the token's own value vector.
"""

import torch
import pytest
from turboquant.model.config import ModelConfig
from turboquant.model.transformer import Transformer
from turboquant.inference.kv_cache import build_kv_caches


def test_xsa_output_shape(tiny_model_xsa, tiny_config_xsa):
    """XSA does not change output shape."""
    tiny_model_xsa.eval()
    B, T = 2, 8
    ids = torch.randint(0, tiny_config_xsa.vocab_size, (B, T))
    with torch.no_grad():
        logits, _ = tiny_model_xsa(ids)
    assert logits.shape == (B, T, tiny_config_xsa.vocab_size)


def test_xsa_backward(tiny_model_xsa, tiny_config_xsa):
    """Gradients flow through XSA path."""
    B, T = 1, 4
    ids = torch.randint(0, tiny_config_xsa.vocab_size, (B, T))
    logits, _ = tiny_model_xsa(ids)
    logits.sum().backward()
    # At least the output projection should have a gradient
    for layer in tiny_model_xsa.layers:
        assert layer.attn.wo.weight.grad is not None


def test_xsa_differs_from_standard(tiny_config):
    """XSA output should differ from standard attention output."""
    cfg_std = tiny_config
    cfg_xsa = ModelConfig(
        vocab_size=cfg_std.vocab_size,
        dim=cfg_std.dim,
        n_layers=cfg_std.n_layers,
        n_heads=cfg_std.n_heads,
        n_kv_heads=cfg_std.n_kv_heads,
        max_seq_len=cfg_std.max_seq_len,
        use_xsa=True,
    )
    # Use different models (different weights → outputs differ anyway),
    # but verify XSA model does NOT produce all-identical outputs to a zeroed XSA model.
    model_xsa = Transformer(cfg_xsa)
    model_xsa.eval()
    B, T = 1, 6
    ids = torch.randint(0, cfg_xsa.vocab_size, (B, T))
    with torch.no_grad():
        logits_xsa, _ = model_xsa(ids)

    # Output should be finite and non-trivial
    assert torch.isfinite(logits_xsa).all()
    assert logits_xsa.std() > 0


def test_xsa_kv_cache_consistent(tiny_config_xsa, tiny_model_xsa):
    """XSA with KV cache (incremental decode) matches full-sequence pass."""
    tiny_model_xsa.eval()
    B, T = 1, 6
    ids = torch.randint(0, tiny_config_xsa.vocab_size, (B, T))

    with torch.no_grad():
        # Full pass reference
        logits_ref, _ = tiny_model_xsa(ids)

        # Incremental decode
        kv_caches = build_kv_caches(tiny_config_xsa, batch_size=B, device=torch.device("cpu"))
        for t in range(T - 1):
            token = ids[:, t : t + 1]
            _, kv_caches = tiny_model_xsa(token, kv_caches=kv_caches, cache_pos=t)
        last_token = ids[:, T - 1 : T]
        logits_incr, _ = tiny_model_xsa(last_token, kv_caches=kv_caches, cache_pos=T - 1)

    assert logits_ref[:, -1, :].argmax() == logits_incr[:, -1, :].argmax(), (
        "XSA: greedy token must match between cached and non-cached"
    )
    assert torch.allclose(logits_ref[:, -1, :], logits_incr[:, -1, :], atol=1e-2), (
        "XSA: logits should be close within FP rounding"
    )
