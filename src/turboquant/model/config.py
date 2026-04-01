"""Model configuration dataclass with presets for each model size."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    # Dimensions
    vocab_size: int = 32000
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4         # GQA: fewer KV heads than query heads
    max_seq_len: int = 1024

    # FFN (computed from dim if not set)
    ffn_hidden_dim: int = 0     # 0 = auto-compute (SwiGLU: 8/3 * dim, rounded to mult of 256)

    # Normalization
    rms_norm_eps: float = 1e-6

    # RoPE
    rope_theta: float = 10000.0

    # Dropout (0 = off for inference; set during training)
    dropout: float = 0.0

    # Attention variant
    use_xsa: bool = False           # Apple XSA: subtract self-value projection from attention output

    # Misc
    tie_embeddings: bool = False    # LLaMA-3 convention: untied

    def __post_init__(self) -> None:
        if self.ffn_hidden_dim == 0:
            # SwiGLU convention: (8/3) * dim, rounded up to multiple of 256
            raw = int(8 * self.dim / 3)
            self.ffn_hidden_dim = math.ceil(raw / 256) * 256

        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        assert self.dim % self.n_heads == 0, (
            f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
        )

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def n_query_groups(self) -> int:
        """Number of query heads per KV head."""
        return self.n_heads // self.n_kv_heads

    @classmethod
    def from_preset(cls, name: str) -> "ModelConfig":
        presets: dict[str, dict[str, Any]] = {
            "turbo-tiny": dict(
                dim=288, n_layers=6, n_heads=6, n_kv_heads=6, max_seq_len=512,
            ),
            "turbo-small": dict(
                dim=768, n_layers=12, n_heads=12, n_kv_heads=4, max_seq_len=1024,
            ),
            "turbo-medium": dict(
                dim=1024, n_layers=24, n_heads=16, n_kv_heads=4, max_seq_len=2048,
            ),
            "turbo-large": dict(
                dim=2048, n_layers=24, n_heads=32, n_kv_heads=8, max_seq_len=2048,
            ),
        }
        if name not in presets:
            raise ValueError(f"Unknown preset '{name}'. Choose from: {list(presets)}")
        return cls(**presets[name])

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        import dataclasses
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False)

    def to_dict(self) -> dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)

    def n_params(self) -> int:
        """Rough parameter count (embedding + all layers + lm_head)."""
        embed = self.vocab_size * self.dim
        attn = self.dim * (
            self.dim                           # wq
            + 2 * self.n_kv_heads * self.head_dim  # wk + wv
            + self.dim                         # wo
        )
        ffn = self.dim * self.ffn_hidden_dim * 3  # w1, w2, w3
        norm = self.dim * 2                     # attn_norm + ffn_norm per layer
        layer = attn + ffn + norm
        lm_head = 0 if self.tie_embeddings else self.vocab_size * self.dim
        return embed + self.n_layers * layer + lm_head
