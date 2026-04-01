"""Token sampling strategies.

All functions operate on logits (pre-softmax scores).
Chain them: apply_temperature → top_k_filter → top_p_filter → sample.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by temperature. temperature=1.0 = no change."""
    if temperature == 0.0:
        # Greedy: set all but max to -inf
        return logits.masked_fill(logits < logits.max(dim=-1, keepdim=True).values, float("-inf"))
    return logits / temperature


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits except the top-k."""
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k, dim=-1)
    min_val = values[..., -1].unsqueeze(-1)
    return logits.masked_fill(logits < min_val, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: keep the smallest set of tokens whose cumulative probability >= p."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens once cumulative prob exceeds p (shift by 1 to keep first token above threshold)
    sorted_remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits = sorted_logits.masked_fill(sorted_remove_mask, float("-inf"))

    # Scatter back to original ordering
    return logits.scatter(-1, sorted_indices, sorted_logits)


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Penalize tokens that have appeared in the context.

    penalty > 1.0 reduces probability of repeated tokens.
    penalty = 1.0 is a no-op.
    """
    if penalty == 1.0:
        return logits
    score = logits.gather(-1, input_ids)
    # Positive logits are divided; negative logits are multiplied
    score = torch.where(score < 0, score * penalty, score / penalty)
    return logits.scatter(-1, input_ids, score)


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    input_ids: torch.Tensor | None = None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """Sample the next token from logits.

    Args:
        logits: (..., vocab_size) unnormalized scores
        temperature: sampling temperature (0 = greedy)
        top_k: keep only top-k tokens (0 = disabled)
        top_p: nucleus filtering (1.0 = disabled)
        input_ids: context tokens for repetition penalty
        repetition_penalty: penalty factor (1.0 = disabled)

    Returns:
        token_id: (...,) sampled token indices
    """
    if input_ids is not None and repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, input_ids, repetition_penalty)

    if temperature == 0.0:
        return logits.argmax(dim=-1)

    logits = apply_temperature(logits, temperature)
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(*probs.shape[:-1])
