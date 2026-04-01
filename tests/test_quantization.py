"""Tests for quantization — output quality and size reduction."""

import torch
import torch.nn as nn
import pytest
from turboquant.model.config import ModelConfig
from turboquant.model.transformer import Transformer
from turboquant.quantization.int8 import Int8Linear, quantize_weight_int8
from turboquant.quantization.int4 import Int4Linear, quantize_weight_int4, pack_int4, unpack_int4
from turboquant.quantization.quantize import quantize_model, model_size_mb


def test_int8_quantize_dequantize():
    w = torch.randn(64, 32) * 2.0
    w_int8, scale = quantize_weight_int8(w)
    assert w_int8.dtype == torch.int8
    w_reconstructed = w_int8.float() * scale.unsqueeze(1)
    cos_sim = torch.nn.functional.cosine_similarity(w.view(1, -1), w_reconstructed.view(1, -1))
    assert cos_sim.item() > 0.99, f"INT8 cosine similarity too low: {cos_sim.item():.4f}"


def test_int8_linear_from_float():
    linear = nn.Linear(128, 64, bias=False)
    q = Int8Linear.from_float(linear)
    x = torch.randn(4, 128)
    y_float = linear(x)
    y_quant = q(x)
    # Outputs should be close (not exact due to quantization)
    cos_sim = torch.nn.functional.cosine_similarity(y_float.view(1, -1), y_quant.view(1, -1))
    assert cos_sim.item() > 0.99


def test_int4_pack_unpack():
    w_int4 = torch.randint(0, 16, (8, 16), dtype=torch.uint8)
    packed = pack_int4(w_int4)
    assert packed.shape == (8, 8)
    unpacked = unpack_int4(packed, 16)
    assert torch.all(unpacked == w_int4)


def test_int4_quantize_roundtrip():
    w = torch.randn(64, 128)
    w_int4, scale, zp = quantize_weight_int4(w, group_size=128)
    from turboquant.quantization.int4 import dequantize_weight_int4
    w_rec = dequantize_weight_int4(w_int4, scale, zp, group_size=128)
    cos_sim = torch.nn.functional.cosine_similarity(w.view(1, -1), w_rec.view(1, -1))
    assert cos_sim.item() > 0.95, f"INT4 cosine similarity too low: {cos_sim.item():.4f}"


def test_quantize_model_int8_reduces_size(tiny_config):
    model_fp = Transformer(tiny_config)
    size_fp = model_size_mb(model_fp)

    model_q = Transformer(tiny_config)
    quantize_model(model_q, method="int8")
    size_q = model_size_mb(model_q)

    assert size_q < size_fp, "Quantized model should be smaller"


def test_quantize_model_forward_still_works(tiny_config):
    model = Transformer(tiny_config)
    quantize_model(model, method="int8")
    ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
    logits, _ = model(ids)
    assert logits.shape == (1, 8, tiny_config.vocab_size)
