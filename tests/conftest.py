"""Shared test fixtures — ultra-tiny config for fast tests."""

import pytest
import torch

from turboquant.model.config import ModelConfig
from turboquant.model.transformer import Transformer


@pytest.fixture
def tiny_config() -> ModelConfig:
    """Ultra-tiny config for fast unit tests."""
    return ModelConfig(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
    )


@pytest.fixture
def tiny_model(tiny_config: ModelConfig) -> Transformer:
    return Transformer(tiny_config)


@pytest.fixture
def tiny_config_xsa() -> ModelConfig:
    """Ultra-tiny config with XSA enabled."""
    return ModelConfig(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        use_xsa=True,
    )


@pytest.fixture
def tiny_model_xsa(tiny_config_xsa: ModelConfig) -> Transformer:
    return Transformer(tiny_config_xsa)


@pytest.fixture
def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
