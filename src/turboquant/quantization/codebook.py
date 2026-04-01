"""Lloyd-Max codebook computation for TurboQuant scalar quantization.

Each coordinate of a random unit-norm vector in high dimensions follows a
Beta distribution that converges to N(0, 1/d). We treat each rotated coordinate
as an independent N(0, 1) scalar (the rotation preserves the distribution up to
scale) and solve the 1-D Lloyd-Max optimal quantization problem.

Codebooks are computed once offline per bit-width and stored as module-level
constants. No training data required.

Reference: TurboQuant (arxiv 2504.19874), Section 3.
"""

from __future__ import annotations

import math
import torch


def _lloyd_max_1d(n_levels: int, n_iter: int = 200, n_samples: int = 200_000) -> torch.Tensor:
    """Solve Lloyd-Max 1-D quantization for a N(0,1) source.

    Returns centroids of shape (n_levels,) sorted in ascending order.
    """
    # Initialize centroids at quantiles of N(0,1)
    quantiles = torch.linspace(0.5 / n_levels, 1.0 - 0.5 / n_levels, n_levels)
    # Inverse CDF of N(0,1) via erfinv
    centroids = math.sqrt(2) * torch.erfinv(2 * quantiles - 1)

    # Draw a large sample from N(0,1) for the expectation approximation
    samples = torch.randn(n_samples)

    for _ in range(n_iter):
        # E-step: assign each sample to nearest centroid
        # dists: (n_samples, n_levels)
        dists = (samples.unsqueeze(1) - centroids.unsqueeze(0)).abs()
        assignments = dists.argmin(dim=1)  # (n_samples,)

        # M-step: update centroids to cluster means
        new_centroids = centroids.clone()
        for k in range(n_levels):
            mask = assignments == k
            if mask.sum() > 0:
                new_centroids[k] = samples[mask].mean()

        # Check convergence
        if (new_centroids - centroids).abs().max() < 1e-8:
            break
        centroids = new_centroids

    return centroids.sort().values


def build_codebooks(max_bits: int = 4) -> dict[int, torch.Tensor]:
    """Return Lloyd-Max codebooks for b = 1 .. max_bits.

    Each codebook is a float32 tensor of shape (2^b,).
    """
    return {b: _lloyd_max_1d(2**b) for b in range(1, max_bits + 1)}


# Pre-computed codebooks for b in {1, 2, 3, 4} — loaded once at import time.
# Each tensor has shape (2^b,) and covers N(0,1) optimally.
_CODEBOOKS: dict[int, torch.Tensor] | None = None


def get_codebook(bits: int) -> torch.Tensor:
    """Return the cached Lloyd-Max codebook for the given bit-width."""
    global _CODEBOOKS
    if _CODEBOOKS is None:
        _CODEBOOKS = build_codebooks(max_bits=4)
    if bits not in _CODEBOOKS:
        raise ValueError(f"bits must be in 1..4, got {bits}")
    return _CODEBOOKS[bits]
