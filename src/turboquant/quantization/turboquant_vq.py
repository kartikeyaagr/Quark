"""TurboQuant vector quantization for KV cache compression.

Two quantizers from the paper (arxiv 2504.19874):

1. TurboQuantMSE — minimises mean squared error.
   Algorithm:
     Quant:   y = Π·x  →  scale to N(0,1)  →  idx_j = argmin_k |y_j - c_k|
     DeQuant: ỹ_j = c[idx_j] · scale  →  x̃ = Πᵀ·ỹ

   The key step: after rotation by an orthogonal Π, each coordinate y_j has
   std ≈ ‖x‖/sqrt(d).  We scale to unit std so the N(0,1) Lloyd-Max codebook
   applies correctly.

2. TurboQuantProd — unbiased inner-product estimation (needed for attention scores).
   Algorithm:
     Quant:   MSE-quantize with (b-1) bits; store residual via QJL sign(S·r) + ‖r‖
     DeQuant: x̃_mse + ‖r‖·(√(π/2)/d)·Sᵀ·sign(S·r)

Both quantizers share a fixed random rotation matrix Π and (for prod) a random
Johnson-Lindenstrauss matrix S, both generated from a fixed seed so they are
deterministic across processes.

Usage:
    tq = TurboQuantMSE(dim=128, bits=3)
    idx, scales = tq.quantize(x)      # x: (..., dim)
    x_hat = tq.dequantize(idx, scales)

    tqp = TurboQuantProd(dim=128, bits=3)
    pkg = tqp.quantize(x)
    x_hat = tqp.dequantize(pkg)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from turboquant.quantization.codebook import get_codebook


def _random_orthogonal(dim: int, seed: int = 42) -> torch.Tensor:
    """Generate a fixed random orthogonal matrix via QR decomposition."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    A = torch.randn(dim, dim, generator=gen)
    Q, _ = torch.linalg.qr(A)
    return Q  # (dim, dim), orthogonal: QᵀQ = I


def _random_gaussian(dim: int, seed: int = 137) -> torch.Tensor:
    """Generate a fixed random Gaussian matrix for QJL."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.randn(dim, dim, generator=gen)  # (dim, dim)


class TurboQuantMSE:
    """Near-optimal MSE vector quantizer.

    After random rotation, each coordinate of x ∈ ℝᵈ has std ≈ ‖x‖/sqrt(d).
    We scale to unit std before applying the N(0,1) Lloyd-Max codebook, storing
    the per-vector scale factor for reconstruction.

    Args:
        dim: vector dimension (head_dim for KV cache)
        bits: bit-width per coordinate (1–4)
    """

    def __init__(self, dim: int, bits: int = 3) -> None:
        if not 1 <= bits <= 4:
            raise ValueError(f"bits must be in 1..4, got {bits}")
        self.dim = dim
        self.bits = bits
        self.n_levels = 2**bits

        # Fixed rotation (orthogonal) and codebook — device-agnostic float32
        self._rotation: torch.Tensor = _random_orthogonal(dim)   # (dim, dim)
        self._codebook: torch.Tensor = get_codebook(bits)         # (n_levels,)

    def _ensure_device(self, ref: torch.Tensor) -> None:
        """Move rotation/codebook to the device/dtype of ref (lazy)."""
        if self._rotation.device != ref.device or self._rotation.dtype != ref.dtype:
            self._rotation = self._rotation.to(device=ref.device, dtype=ref.dtype)
            self._codebook = self._codebook.to(device=ref.device, dtype=ref.dtype)

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize vectors.

        Args:
            x: (..., dim) — any leading batch dims

        Returns:
            indices: (..., dim) uint8 — codebook index per coordinate (max 4 bits → fits uint8)
            scales: (...,) float — per-vector scale = ‖Π·x‖ / sqrt(dim), used for reconstruction
        """
        self._ensure_device(x)
        shape = x.shape
        x_flat = x.reshape(-1, self.dim).float()  # (N, dim)

        # Random rotation: y = Π·x  →  (N, dim)
        y = x_flat @ self._rotation.T  # shape (N, dim)

        # Per-vector scale: std of each coordinate ≈ ‖y‖/sqrt(dim)
        scales = y.norm(dim=-1) / math.sqrt(self.dim)  # (N,)

        # Scale to unit std for the N(0,1) codebook
        y_norm = y / scales.unsqueeze(-1).clamp(min=1e-8)  # (N, dim) ~ N(0,1) per coord

        # Coordinate-wise nearest-centroid lookup
        dists = (y_norm.unsqueeze(-1) - self._codebook.unsqueeze(0).unsqueeze(0)).abs()
        indices = dists.argmin(dim=-1).to(torch.uint8)  # (N, dim)

        return indices.reshape(*shape[:-1], self.dim), scales.reshape(*shape[:-1])

    def dequantize(self, indices: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Reconstruct vectors from quantized indices.

        Args:
            indices: (..., dim) uint8
            scales: (...,) per-vector scale factors

        Returns:
            x_hat: (..., dim) reconstructed vectors
        """
        cb = self._codebook.to(device=scales.device, dtype=scales.dtype)
        rotation = self._rotation.to(device=scales.device, dtype=scales.dtype)

        shape = indices.shape
        idx_flat = indices.reshape(-1, self.dim).long()
        scales_flat = scales.reshape(-1)  # (N,)

        # Lookup centroids per coordinate  →  (N, dim)
        y_norm_hat = cb[idx_flat]

        # Re-scale back from unit std
        y_hat = y_norm_hat * scales_flat.unsqueeze(-1)  # (N, dim)

        # Inverse rotation: x̃ = Πᵀ·ŷ
        x_hat = y_hat @ rotation  # (N, dim)

        return x_hat.reshape(*shape[:-1], self.dim).to(scales.dtype)


@dataclass
class TurboQuantProdPackage:
    """Container for TurboQuantProd compressed representation."""
    mse_indices: torch.Tensor    # (..., dim) uint8 — (b-1)-bit MSE indices
    qjl_packed: torch.Tensor     # (..., dim//8) uint8 — sign(S·r) bit-packed, 8 signs/byte
    residual_norms: torch.Tensor # (...,) float  — ‖r‖₂
    mse_scales: torch.Tensor     # (...,) float  — MSE scale factors


def _pack_signs(signs: torch.Tensor) -> torch.Tensor:
    """Bit-pack sign tensor into uint8: 8 signs per byte (LSB=sign[0]).

    Args:
        signs: (..., D) int8 with values ±1 (or 0 treated as +1)
    Returns:
        packed: (..., D//8) uint8
    """
    shape = signs.shape
    D = shape[-1]
    assert D % 8 == 0, f"dim must be divisible by 8 for bit-packing, got {D}"
    flat = ((signs.reshape(-1, D) > 0).to(torch.uint8))  # (N, D) as 0/1
    # Pack 8 consecutive bits into one uint8 byte
    N = flat.shape[0]
    flat = flat.reshape(N, D // 8, 8)  # (N, D//8, 8)
    weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128],
                           dtype=torch.uint8, device=flat.device)
    packed = (flat * weights).sum(dim=-1).to(torch.uint8)  # (N, D//8)
    return packed.reshape(*shape[:-1], D // 8)


def _unpack_signs(packed: torch.Tensor, D: int) -> torch.Tensor:
    """Unpack uint8 bit-packed tensor back to ±1 int8.

    Args:
        packed: (..., D//8) uint8
        D: original dimension
    Returns:
        signs: (..., D) float — values ±1.0
    """
    shape = packed.shape
    N = packed.reshape(-1, D // 8).shape[0]
    flat = packed.reshape(N, D // 8)
    # Expand each byte into 8 bits
    bits = torch.zeros(N, D, dtype=torch.float32, device=packed.device)
    for i in range(8):
        bits[:, i::8] = ((flat >> i) & 1).float()  # extract bit i from each byte
    # Map 0 → -1, 1 → +1
    signs = bits * 2 - 1  # (N, D)
    return signs.reshape(*shape[:-1], D)


class TurboQuantProd:
    """Unbiased inner-product vector quantizer (TurboQuant_prod).

    Guarantees: 𝔼[⟨y, x̃⟩] = ⟨y, x⟩  for any query y.

    Uses (b-1)-bit MSE quantization plus 1-bit QJL on the residual.
    The combined estimator is unbiased for inner products.

    Args:
        dim: vector dimension (must be divisible by 8 for bit-packing)
        bits: effective bit-width (2–4); MSE uses (bits-1), QJL uses 1 bit
    """

    def __init__(self, dim: int, bits: int = 3) -> None:
        if not 2 <= bits <= 4:
            raise ValueError(f"bits must be in 2..4 for TurboQuantProd, got {bits}")
        if dim % 8 != 0:
            raise ValueError(f"dim must be divisible by 8 for QJL packing, got {dim}")
        self.dim = dim
        self.bits = bits

        self._mse = TurboQuantMSE(dim=dim, bits=bits - 1)
        self._S: torch.Tensor = _random_gaussian(dim)  # (dim, dim) QJL matrix

    def _S_on(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._S.device != device or self._S.dtype != dtype:
            self._S = self._S.to(device=device, dtype=dtype)
        return self._S

    def quantize(self, x: torch.Tensor) -> TurboQuantProdPackage:
        """Quantize for inner-product preservation.

        Args:
            x: (..., dim)

        Returns:
            TurboQuantProdPackage
        """
        # Step 1: MSE quantize at (b-1) bits
        mse_idx, mse_scales = self._mse.quantize(x)
        x_mse = self._mse.dequantize(mse_idx, mse_scales)

        # Step 2: residual
        r = x.float() - x_mse.float()
        r_norms = r.reshape(-1, self.dim).norm(dim=-1)  # (N,)

        # Step 3: QJL — sign(S·r), bit-packed
        S = self._S_on(x.device, torch.float32)
        shape = x.shape
        r_flat = r.reshape(-1, self.dim)   # (N, dim)
        Sr = r_flat @ S.T                  # (N, dim)
        signs = Sr.sign().to(torch.int8)   # (N, dim) ±1
        # Treat 0 (zero exactly) as +1
        signs = signs.reshape(*shape[:-1], self.dim)
        qjl_packed = _pack_signs(signs)    # (..., dim//8) uint8

        r_norms = r_norms.reshape(*shape[:-1])

        return TurboQuantProdPackage(
            mse_indices=mse_idx,
            qjl_packed=qjl_packed,
            residual_norms=r_norms,
            mse_scales=mse_scales,
        )

    def dequantize(self, pkg: TurboQuantProdPackage) -> torch.Tensor:
        """Reconstruct with unbiased inner-product estimator.

        Returns:
            x_hat: (..., dim)
        """
        # MSE reconstruction
        x_mse = self._mse.dequantize(pkg.mse_indices, pkg.mse_scales)

        # QJL residual reconstruction: ‖r‖ · (√(π/2)/d) · Sᵀ·sign(S·r)
        S = self._S_on(x_mse.device, x_mse.dtype)
        signs = _unpack_signs(pkg.qjl_packed, self.dim).to(x_mse.dtype)  # (..., dim)

        shape = signs.shape
        signs_flat = signs.reshape(-1, self.dim)
        r_norms_flat = pkg.residual_norms.reshape(-1).to(x_mse.dtype)

        # Sᵀ·sign(S·r): (N, dim)
        St_signs = signs_flat @ S  # (N, dim)

        # Scale: ‖r‖ · √(π/2) / d
        scale = r_norms_flat * math.sqrt(math.pi / 2) / self.dim  # (N,)
        r_hat = St_signs * scale.unsqueeze(-1)  # (N, dim)
        r_hat = r_hat.reshape(*shape[:-1], self.dim)

        return (x_mse + r_hat).to(x_mse.dtype)
