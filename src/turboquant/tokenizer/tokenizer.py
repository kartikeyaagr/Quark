"""TurboTokenizer — thin wrapper around HuggingFace tokenizers.

Provides a clean interface with automatic BOS/EOS injection,
batch encoding, and padding.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer

from turboquant.tokenizer.special_tokens import (
    BOS_ID, EOS_ID, PAD_ID,
    BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN,
)


class TurboTokenizer:
    def __init__(self, hf_tokenizer: HFTokenizer) -> None:
        self._tok = hf_tokenizer
        self._tok.enable_padding(pad_id=PAD_ID, pad_token=PAD_TOKEN)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> "TurboTokenizer":
        tok = HFTokenizer.from_file(str(Path(path) / "tokenizer.json"))
        return cls(tok)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def pad_id(self) -> int:
        return PAD_ID

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode a single string to token IDs."""
        ids: list[int] = self._tok.encode(text).ids
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode token IDs back to a string."""
        return self._tok.decode(ids, skip_special_tokens=skip_special)

    def batch_encode(
        self,
        texts: list[str],
        add_bos: bool = True,
        add_eos: bool = False,
        pad: bool = True,
    ) -> list[list[int]]:
        """Encode a batch of strings."""
        encodings = self._tok.encode_batch(texts)
        result = []
        for enc in encodings:
            ids: list[int] = enc.ids
            if add_bos:
                ids = [BOS_ID] + ids
            if add_eos:
                ids = ids + [EOS_ID]
            result.append(ids)

        if pad:
            max_len = max(len(ids) for ids in result)
            result = [ids + [PAD_ID] * (max_len - len(ids)) for ids in result]

        return result

    def batch_decode(self, batch: list[list[int]], skip_special: bool = True) -> list[str]:
        return self._tok.decode_batch(batch, skip_special_tokens=skip_special)

    def __len__(self) -> int:
        return self.vocab_size
