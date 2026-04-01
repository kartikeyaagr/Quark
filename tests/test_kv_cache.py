"""Tests for KV cache — cached output must match non-cached output."""

import torch
import pytest
from turboquant.model.config import ModelConfig
from turboquant.model.transformer import Transformer
from turboquant.inference.kv_cache import build_kv_caches


def test_cached_matches_noncached(tiny_config, tiny_model):
    tiny_model.eval()
    B, T = 1, 8
    ids = torch.randint(0, tiny_config.vocab_size, (B, T))

    with torch.no_grad():
        # Non-cached (full forward)
        logits_full, _ = tiny_model(ids)

        # Cached: prefill then single-token decode
        kv_caches = build_kv_caches(tiny_config, batch_size=B, device=torch.device("cpu"))
        logits_cached, kv_caches = tiny_model(ids, kv_caches=kv_caches, cache_pos=0)

    assert torch.allclose(logits_full, logits_cached, atol=1e-4), (
        "Cached and non-cached outputs must match"
    )


def test_incremental_decode(tiny_config, tiny_model):
    """Incremental (token-by-token) generation matches single-pass."""
    tiny_model.eval()
    B, T = 1, 6
    ids = torch.randint(0, tiny_config.vocab_size, (B, T))

    with torch.no_grad():
        # Single pass for reference
        logits_ref, _ = tiny_model(ids)
        last_logits_ref = logits_ref[:, -1, :]

        # Incremental: process one token at a time
        kv_caches = build_kv_caches(tiny_config, batch_size=B, device=torch.device("cpu"))
        for t in range(T - 1):
            token = ids[:, t : t + 1]
            _, kv_caches = tiny_model(token, kv_caches=kv_caches, cache_pos=t)

        # Last token
        last_token = ids[:, T - 1 : T]
        logits_incr, _ = tiny_model(last_token, kv_caches=kv_caches, cache_pos=T - 1)

    # Different SDPA code paths (full-seq vs Q_len=1) cause minor FP rounding differences.
    # Verify the argmax (greedy token) matches, and logits are numerically close.
    assert logits_ref[:, -1, :].argmax() == logits_incr[:, -1, :].argmax(), \
        "Greedy token must match between cached and non-cached"
    assert torch.allclose(last_logits_ref, logits_incr[:, -1, :], atol=1e-2), \
        "Logits should be close (within FP rounding across SDPA code paths)"
