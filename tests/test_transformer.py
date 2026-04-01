"""Tests for the full Transformer model."""

import torch
import pytest
from turboquant.model.config import ModelConfig
from turboquant.model.transformer import Transformer


def test_forward_shape(tiny_model, tiny_config):
    B, T = 2, 16
    ids = torch.randint(0, tiny_config.vocab_size, (B, T))
    logits, caches = tiny_model(ids)
    assert logits.shape == (B, T, tiny_config.vocab_size)
    assert len(caches) == tiny_config.n_layers


def test_backward_pass(tiny_model, tiny_config):
    ids = torch.randint(0, tiny_config.vocab_size, (2, 8))
    logits, _ = tiny_model(ids)
    loss = logits.mean()
    loss.backward()
    for p in tiny_model.parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad for param shape {p.shape}"


def test_overfit_single_batch(tiny_config):
    """Model should overfit to a single batch (loss → near zero)."""
    model = Transformer(tiny_config)
    import torch.nn.functional as F
    from turboquant.training.optimizer import configure_optimizer

    optimizer = configure_optimizer(model, lr=1e-2, fused=False)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 16))
    labels = torch.roll(ids, -1, dims=1)

    initial_loss = None
    for step in range(200):
        logits, _ = model(ids)
        loss = F.cross_entropy(logits.view(-1, tiny_config.vocab_size), labels.view(-1))
        if initial_loss is None:
            initial_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.item() < initial_loss * 0.1, (
        f"Expected loss to decrease by 10x, but {initial_loss:.4f} → {loss.item():.4f}"
    )


def test_param_count(tiny_config):
    model = Transformer(tiny_config)
    actual = model.n_params()
    expected = tiny_config.n_params()
    # Allow small discrepancy due to norms
    assert abs(actual - expected) / expected < 0.02


def test_sequence_length_assertion(tiny_model, tiny_config):
    ids = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.max_seq_len + 1))
    with pytest.raises(AssertionError):
        tiny_model(ids)
