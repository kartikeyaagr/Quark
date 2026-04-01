"""Rotary Position Embeddings (RoPE).

Applied to query and key tensors only. Uses complex number multiplication
for efficient rotation. Compatible with GQA — computed once, shared.
"""

import torch


def precompute_freqs_cis(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Precompute complex-valued rotation frequencies.

    Returns: (max_seq_len, head_dim // 2) complex tensor.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    # Frequencies: theta^(-2i/d) for i in [0, d/2)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device)
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim // 2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.

    Args:
        xq: (batch, seq_len, n_heads, head_dim)
        xk: (batch, seq_len, n_kv_heads, head_dim)
        freqs_cis: (seq_len, head_dim // 2) complex

    Returns: rotated xq, xk with same shapes.
    """
    # Reshape to complex: (batch, seq_len, heads, head_dim // 2)
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # freqs_cis: (seq_len, head_dim // 2) → broadcast over batch and heads
    freqs = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim // 2)

    xq_out = torch.view_as_real(xq_c * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_c * freqs).flatten(-2)

    return xq_out.to(xq.dtype), xk_out.to(xk.dtype)
