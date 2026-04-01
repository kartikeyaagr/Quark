"""Train a byte-level BPE tokenizer using HuggingFace tokenizers (Rust-backed).

Usage:
    trainer = BPETrainer(vocab_size=32000)
    trainer.train(["path/to/corpus1.txt", "path/to/corpus2.txt"])
    trainer.save("tokenizers/mytok")
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import models, pre_tokenizers, trainers, decoders

from turboquant.tokenizer.special_tokens import (
    BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, SPECIAL_TOKENS
)


class BPETrainer:
    """Train a byte-level BPE tokenizer."""

    def __init__(
        self,
        vocab_size: int = 32000,
        min_frequency: int = 2,
    ) -> None:
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

    def _build_tokenizer(self) -> HFTokenizer:
        tokenizer = HFTokenizer(models.BPE(unk_token=UNK_TOKEN))

        # Byte-level pre-tokenizer: handles all unicode, no UNK from encoding
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # type: ignore
        tokenizer.decoder = decoders.ByteLevel()  # type: ignore

        return tokenizer

    def train(
        self,
        files: list[str] | list[Path],
        save_path: str | Path | None = None,
    ) -> HFTokenizer:
        """Train on a list of text files and optionally save."""
        tokenizer = self._build_tokenizer()

        trainer = trainers.BpeTrainer(  # type: ignore
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
        )

        str_files = [str(f) for f in files]
        tokenizer.train(str_files, trainer)

        if save_path is not None:
            self.save(tokenizer, save_path)

        return tokenizer

    def train_from_iterator(
        self,
        iterator: Iterator[str],
        save_path: str | Path | None = None,
        length: int | None = None,
    ) -> HFTokenizer:
        """Train from a string iterator (useful for in-memory datasets)."""
        tokenizer = self._build_tokenizer()

        trainer = trainers.BpeTrainer(  # type: ignore
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
        )

        tokenizer.train_from_iterator(iterator, trainer, length=length)

        if save_path is not None:
            self.save(tokenizer, save_path)

        return tokenizer

    @staticmethod
    def save(tokenizer: HFTokenizer, path: str | Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(Path(path) / "tokenizer.json"))

    @staticmethod
    def load(path: str | Path) -> HFTokenizer:
        return HFTokenizer.from_file(str(Path(path) / "tokenizer.json"))
