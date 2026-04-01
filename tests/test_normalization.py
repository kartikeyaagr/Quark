"""Tests for RMSNorm."""

import torch
import pytest
from turboquant.model.normalization import RMSNorm


def test_output_shape():
    norm = RMSNorm(64)
    x = torch.randn(2, 10, 64)
    assert norm(x).shape == x.shape


def test_unit_rms():
    """After normalization, RMS of output should be close to 1 (before weight scaling)."""
    norm = RMSNorm(64)
    # Init weight to ones — output RMS should equal weight (1.0)
    x = torch.randn(4, 16, 64) * 10  # large magnitude
    y = norm(x)
    rms = y.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.01)


def test_preserves_dtype():
    norm = RMSNorm(32)
    x = torch.randn(2, 8, 32).to(torch.bfloat16)
    assert norm(x).dtype == torch.bfloat16


def test_gradient_flows():
    norm = RMSNorm(32)
    x = torch.randn(2, 4, 32, requires_grad=True)
    y = norm(x)
    y.sum().backward()
    assert x.grad is not None
    assert norm.weight.grad is not None
