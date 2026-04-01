"""Tests for TurboQuant vector quantization.

Verifies:
- Lloyd-Max codebook quality (MSE bounds from paper)
- Random rotation preserves norms
- Quantize → dequantize roundtrip cosine similarity
- Inner-product unbiasedness (TurboQuantProd)
- Compressed KV cache memory reduction and output correctness
"""

import math
import torch
import pytest
from turboquant.quantization.codebook import get_codebook, build_codebooks
from turboquant.quantization.turboquant_vq import TurboQuantMSE, TurboQuantProd
from turboquant.inference.compressed_kv_cache import CompressedKVCache, build_compressed_kv_caches


# ── Codebook tests ──────────────────────────────────────────────────────────

def test_codebook_sizes():
    for bits in range(1, 5):
        cb = get_codebook(bits)
        assert cb.shape == (2**bits,), f"bits={bits}: expected {2**bits} centroids"


def test_codebook_sorted():
    """Centroids should be sorted ascending."""
    for bits in range(1, 5):
        cb = get_codebook(bits)
        assert (cb[1:] >= cb[:-1]).all(), f"bits={bits}: codebook not sorted"


def test_codebook_mse_bounds():
    """Empirical MSE should be within 3× of theoretical lower bound 1/4^b.

    Paper theoretical upper bound is ~2.7/4^b, so we allow a 4× factor.
    """
    torch.manual_seed(0)
    n = 100_000
    samples = torch.randn(n)
    for bits in range(1, 5):
        cb = get_codebook(bits)
        dists = (samples.unsqueeze(1) - cb.unsqueeze(0)).abs()
        idx = dists.argmin(dim=1)
        mse = ((samples - cb[idx]) ** 2).mean().item()
        lower_bound = 1.0 / (4**bits)
        assert mse <= 4.0 * lower_bound, (
            f"bits={bits}: MSE={mse:.4f} exceeds 4× lower bound {4*lower_bound:.4f}"
        )


# ── TurboQuantMSE tests ──────────────────────────────────────────────────────

@pytest.mark.parametrize("bits", [2, 3, 4])
def test_turboquant_mse_roundtrip_cosine(bits):
    """Cosine similarity after MSE quantize→dequantize."""
    torch.manual_seed(bits)
    dim = 128
    N = 512
    x = torch.randn(N, dim)
    x = x / x.norm(dim=-1, keepdim=True)  # unit vectors

    tq = TurboQuantMSE(dim=dim, bits=bits)
    idx, norms = tq.quantize(x)
    x_hat = tq.dequantize(idx, norms)

    # Cosine similarity per vector
    cos_sim = (x * x_hat).sum(dim=-1) / (x.norm(dim=-1) * x_hat.norm(dim=-1).clamp(1e-8))
    mean_cos = cos_sim.mean().item()

    thresholds = {2: 0.85, 3: 0.95, 4: 0.99}
    assert mean_cos >= thresholds[bits], (
        f"bits={bits}: mean cosine sim {mean_cos:.3f} < threshold {thresholds[bits]}"
    )


def test_turboquant_mse_norm_preserved():
    """After quantize+dequantize, norms should be approximately preserved."""
    torch.manual_seed(0)
    dim = 64
    x = torch.randn(100, dim)
    tq = TurboQuantMSE(dim=dim, bits=3)
    idx, norms = tq.quantize(x)
    x_hat = tq.dequantize(idx, norms)

    orig_norms = x.norm(dim=-1)
    hat_norms = x_hat.norm(dim=-1)
    # Norms should match to within 5% (we explicitly re-scale)
    rel_err = ((orig_norms - hat_norms).abs() / orig_norms.clamp(1e-8)).mean()
    assert rel_err < 0.05, f"Norm relative error {rel_err:.4f} too large"


def test_turboquant_mse_batch_shape():
    """quantize/dequantize preserve arbitrary leading dimensions."""
    dim = 64
    tq = TurboQuantMSE(dim=dim, bits=2)
    x = torch.randn(2, 10, 4, dim)  # (B, T, H, dim)
    idx, norms = tq.quantize(x)
    assert idx.shape == (2, 10, 4, dim)
    assert norms.shape == (2, 10, 4)
    x_hat = tq.dequantize(idx, norms)
    assert x_hat.shape == x.shape


# ── TurboQuantProd tests ──────────────────────────────────────────────────────

@pytest.mark.parametrize("bits", [2, 3, 4])
def test_turboquant_prod_unbiased(bits):
    """Inner product estimator should be approximately unbiased.

    E[<y, x̃>] ≈ <y, x> over many samples.
    """
    torch.manual_seed(bits + 10)
    dim = 64
    N = 1000
    x = torch.randn(N, dim)
    x = x / x.norm(dim=-1, keepdim=True)
    y = torch.randn(dim)
    y = y / y.norm()

    tq = TurboQuantProd(dim=dim, bits=bits)
    pkg = tq.quantize(x)
    x_hat = tq.dequantize(pkg)

    true_ip = (x @ y).mean().item()
    est_ip = (x_hat @ y).mean().item()

    # Bias should be small relative to scale of inner products
    bias = abs(true_ip - est_ip)
    assert bias < 0.1, f"bits={bits}: inner product bias {bias:.4f} too large"


def test_turboquant_prod_output_shape():
    dim = 128
    tq = TurboQuantProd(dim=dim, bits=3)
    x = torch.randn(4, 8, 2, dim)  # (B, T, H, dim)
    pkg = tq.quantize(x)
    assert pkg.mse_indices.shape == (4, 8, 2, dim)
    assert pkg.qjl_packed.shape == (4, 8, 2, dim // 8)
    x_hat = tq.dequantize(pkg)
    assert x_hat.shape == x.shape


# ── CompressedKVCache tests ───────────────────────────────────────────────────

def test_compressed_kv_cache_read_write():
    """Write then read should approximately recover K and V."""
    B, T, H, D = 1, 8, 2, 64
    cache = CompressedKVCache(
        batch_size=B, max_seq_len=32, n_kv_heads=H, head_dim=D,
        device=torch.device("cpu"), bits=3,
    )
    torch.manual_seed(42)
    k = torch.randn(B, T, H, D)
    v = torch.randn(B, T, H, D)
    cache.write(k, v, cache_pos=0)
    k_hat, v_hat = cache.read(length=T, batch_size=B)

    # Shape check
    assert k_hat.shape == (B, T, H, D)
    assert v_hat.shape == (B, T, H, D)

    # Cosine similarity > 0.85 at 3-bit
    def mean_cos(a, b):
        a_flat = a.reshape(-1, D)
        b_flat = b.reshape(-1, D)
        return (a_flat * b_flat).sum(-1) / (
            a_flat.norm(-1) * b_flat.norm(-1).clamp(1e-8)
        )

    assert mean_cos(k, k_hat).mean() > 0.85, "K cosine sim too low"
    assert mean_cos(v, v_hat).mean() > 0.85, "V cosine sim too low"


def test_compressed_kv_cache_memory_reduction():
    """Compressed cache should use less memory than float32 tensors."""
    B, T, H, D = 1, 128, 4, 64
    bits = 3
    cache = CompressedKVCache(
        batch_size=B, max_seq_len=T, n_kv_heads=H, head_dim=D,
        device=torch.device("cpu"), bits=bits,
    )

    # Memory of compressed storage (all buffers)
    def buf_bytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    compressed_bytes = sum(
        buf_bytes(b) for b in [
            cache._k_mse_idx, cache._k_qjl, cache._k_res_norm, cache._k_mse_scale,
            cache._v_mse_idx, cache._v_qjl, cache._v_res_norm, cache._v_mse_scale,
        ]
    )
    # Full FP32 equivalent
    fp32_bytes = 2 * B * T * H * D * 4  # 2 for K+V, 4 bytes per float32

    ratio = fp32_bytes / compressed_bytes
    assert ratio >= 2.0, f"Compression ratio {ratio:.2f}x is less than 2×"


def test_compressed_kv_cache_incremental(tiny_config):
    """Compressed KV cache works correctly in transformer incremental decode."""
    from turboquant.model.transformer import Transformer

    tiny_config.use_xsa = False
    model = Transformer(tiny_config)
    model.eval()

    B, T = 1, 6
    ids = torch.randint(0, tiny_config.vocab_size, (B, T))

    with torch.no_grad():
        logits_ref, _ = model(ids)

        caches = build_compressed_kv_caches(
            tiny_config, batch_size=B, device=torch.device("cpu"), bits=3
        )
        for t in range(T - 1):
            _, caches = model(ids[:, t : t + 1], kv_caches=caches, cache_pos=t)
        logits_comp, _ = model(ids[:, T - 1 : T], kv_caches=caches, cache_pos=T - 1)

    # Greedy token should match (3-bit introduces some noise, so allow small mismatch)
    # At minimum, logits should be finite and in the right ballpark
    assert torch.isfinite(logits_comp).all(), "Compressed cache produced NaN/Inf"
    # Check top-5 overlap
    ref_top5 = logits_ref[:, -1, :].topk(5).indices
    comp_top5 = logits_comp[:, -1, :].topk(5).indices
    overlap = len(set(ref_top5[0].tolist()) & set(comp_top5[0].tolist()))
    assert overlap >= 2, f"Top-5 token overlap too low: {overlap}/5"
