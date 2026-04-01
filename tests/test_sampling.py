"""Tests for sampling strategies."""

import torch
import pytest
from turboquant.inference.sampling import (
    apply_temperature, top_k_filter, top_p_filter,
    apply_repetition_penalty, sample
)


def test_temperature_greedy():
    logits = torch.tensor([[1.0, 5.0, 3.0]])
    result = apply_temperature(logits, 0.0)
    # All non-max values should be -inf
    assert result[0, 1] == logits[0, 1]  # max preserved
    assert result[0, 0] == float("-inf")
    assert result[0, 2] == float("-inf")


def test_top_k_keeps_k_tokens():
    logits = torch.randn(1, 100)
    filtered = top_k_filter(logits, k=10)
    finite_count = (filtered != float("-inf")).sum().item()
    assert finite_count == 10


def test_top_p_sums_to_at_least_p():
    logits = torch.randn(1, 100)
    p = 0.9
    filtered = top_p_filter(logits, p=p)
    probs = torch.softmax(filtered, dim=-1)
    total_prob = probs[probs > 0].sum().item()
    assert total_prob >= p


def test_repetition_penalty_reduces_repeated():
    # Use positive logits — zero / penalty = 0, so penalty needs non-zero input
    logits = torch.ones(1, 10)
    input_ids = torch.tensor([[3, 5, 7]])  # these tokens should be penalized
    penalized = apply_repetition_penalty(logits.clone(), input_ids, penalty=1.5)
    for idx in [3, 5, 7]:
        assert penalized[0, idx] < logits[0, idx]
    # Non-repeated tokens should be unchanged
    for idx in [0, 1, 2]:
        assert penalized[0, idx] == logits[0, idx]


def test_sample_greedy_deterministic():
    logits = torch.tensor([[0.0, 10.0, 0.0]])  # token 1 dominates
    result = sample(logits, temperature=0.0)
    assert result.item() == 1


def test_sample_shape():
    logits = torch.randn(4, 100)
    result = sample(logits, temperature=1.0, top_k=20)
    assert result.shape == (4,)
    assert all(0 <= r < 100 for r in result.tolist())
