"""Tests for Rotary Position Embeddings."""

import torch
import pytest
from turboquant.model.rope import precompute_freqs_cis, apply_rotary_emb


def test_freqs_shape():
    freqs = precompute_freqs_cis(head_dim=64, max_seq_len=128)
    assert freqs.shape == (128, 32)  # (seq, head_dim // 2)
    assert freqs.is_complex()


def test_apply_rotary_shape():
    B, T, H, D = 2, 8, 4, 64
    xq = torch.randn(B, T, H, D)
    xk = torch.randn(B, T, 2, D)
    freqs = precompute_freqs_cis(D, T)
    xq_rot, xk_rot = apply_rotary_emb(xq, xk, freqs)
    assert xq_rot.shape == xq.shape
    assert xk_rot.shape == xk.shape


def test_preserves_magnitude():
    """RoPE is an isometric rotation — it should preserve vector magnitude."""
    B, T, H, D = 1, 4, 2, 32
    xq = torch.randn(B, T, H, D)
    freqs = precompute_freqs_cis(D, T)
    xq_rot, _ = apply_rotary_emb(xq, xq.clone(), freqs)
    orig_norm = xq.float().norm(dim=-1)
    rot_norm = xq_rot.float().norm(dim=-1)
    assert torch.allclose(orig_norm, rot_norm, atol=1e-5)


def test_different_positions_give_different_rotations():
    B, H, D = 1, 1, 32
    xq = torch.ones(B, 2, H, D)
    freqs = precompute_freqs_cis(D, 4)
    xq_rot, _ = apply_rotary_emb(xq, xq.clone(), freqs[:2])
    # Positions 0 and 1 should yield different rotations
    assert not torch.allclose(xq_rot[:, 0], xq_rot[:, 1])
