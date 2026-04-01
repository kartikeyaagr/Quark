"""Tests for ModelConfig."""

import math
import pytest
from turboquant.model.config import ModelConfig


def test_ffn_hidden_dim_auto():
    cfg = ModelConfig(dim=768, n_layers=1, n_heads=4, n_kv_heads=4)
    # 8/3 * 768 = 2048, already a multiple of 256
    assert cfg.ffn_hidden_dim == 2048
    assert cfg.ffn_hidden_dim % 256 == 0


def test_head_dim():
    cfg = ModelConfig(dim=512, n_layers=1, n_heads=8, n_kv_heads=4)
    assert cfg.head_dim == 64


def test_n_query_groups():
    cfg = ModelConfig(dim=512, n_layers=1, n_heads=8, n_kv_heads=2)
    assert cfg.n_query_groups == 4


def test_invalid_heads():
    with pytest.raises(AssertionError):
        ModelConfig(dim=512, n_layers=1, n_heads=7, n_kv_heads=7)


def test_presets():
    for name in ["turbo-tiny", "turbo-small", "turbo-medium", "turbo-large"]:
        cfg = ModelConfig.from_preset(name)
        assert cfg.n_heads % cfg.n_kv_heads == 0
        assert cfg.dim % cfg.n_heads == 0

    with pytest.raises(ValueError):
        ModelConfig.from_preset("does-not-exist")


def test_param_count_roughly_correct():
    cfg = ModelConfig.from_preset("turbo-tiny")
    count = cfg.n_params()
    assert 10e6 < count < 40e6, f"Expected ~15M params, got {count/1e6:.1f}M"
