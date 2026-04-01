"""Autoregressive text generator with KV-cache.

Two modes:
  - generate(): returns the full token list at once
  - stream(): yields tokens one at a time (for streaming output)
"""

from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn

from turboquant.model.config import ModelConfig
from turboquant.inference.kv_cache import build_kv_caches
from turboquant.inference.sampling import sample
from turboquant.tokenizer.tokenizer import TurboTokenizer
from turboquant.tokenizer.special_tokens import EOS_ID


class Generator:
    def __init__(
        self,
        model: nn.Module,
        config: ModelConfig,
        tokenizer: TurboTokenizer,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_on_eos: bool = True,
    ) -> str:
        """Generate text from a prompt string. Returns the generated text only."""
        tokens = list(self._generate_tokens(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_on_eos=stop_on_eos,
        ))
        return self.tokenizer.decode(tokens)

    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_on_eos: bool = True,
    ) -> Iterator[str]:
        """Stream generated tokens one at a time as decoded strings."""
        for token_id in self._generate_tokens(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_on_eos=stop_on_eos,
        ):
            yield self.tokenizer.decode([token_id])

    @torch.inference_mode()
    def _generate_tokens(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        stop_on_eos: bool,
    ) -> Iterator[int]:
        # Encode prompt
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        B, T = input_ids.shape

        # Allocate KV caches
        kv_caches = build_kv_caches(self.config, batch_size=B, device=self.device)

        # Prefill: process the entire prompt at once
        logits, kv_caches = self.model(input_ids, kv_caches=kv_caches, cache_pos=0)
        cache_pos = T

        # Take logits for the last prompt token
        next_token_logits = logits[:, -1, :]  # (B, vocab)

        context_ids = input_ids  # track all ids for repetition penalty

        for _ in range(max_new_tokens):
            next_token = sample(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                input_ids=context_ids.view(1, -1),
                repetition_penalty=repetition_penalty,
            )  # (B,)

            token_id = next_token.item()

            if stop_on_eos and token_id == EOS_ID:
                break

            yield int(token_id)

            # Decode step: feed only the new token
            next_input = next_token.unsqueeze(1)  # (B, 1)
            context_ids = torch.cat([context_ids, next_input], dim=1)

            logits, kv_caches = self.model(
                next_input, kv_caches=kv_caches, cache_pos=cache_pos
            )
            cache_pos += 1

            if cache_pos >= self.config.max_seq_len:
                break  # exceeded context window

            next_token_logits = logits[:, -1, :]
