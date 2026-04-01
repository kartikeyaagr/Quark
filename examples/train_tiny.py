"""Quick example: train turbo-tiny on repeated phrases.

Run:  uv run python examples/train_tiny.py
"""

import torch
from torch.utils.data import DataLoader

from turboquant.model.config import ModelConfig
from turboquant.model.transformer import Transformer
from turboquant.tokenizer.trainer import BPETrainer
from turboquant.tokenizer.tokenizer import TurboTokenizer
from turboquant.data.dataset import InMemoryDataset
from turboquant.data.collator import CausalLMCollator
from turboquant.training.trainer import Trainer, TrainingConfig


def main() -> None:
    # 1. Build a tiny in-memory tokenizer from repeated phrases
    texts = ["hello world " * 20, "the quick brown fox " * 20, "to be or not to be " * 20]

    trainer_tok = BPETrainer(vocab_size=512, min_frequency=1)
    hf_tok = trainer_tok.train_from_iterator(iter(texts))

    tokenizer = TurboTokenizer(hf_tok)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # 2. Build a tiny model
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        dim=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
    )
    model = Transformer(config)
    print(f"Model: {model.n_params() / 1e6:.2f}M params")

    # 3. Dataset + DataLoader
    dataset = InMemoryDataset(texts * 50, tokenizer, seq_len=config.max_seq_len)
    loader = DataLoader(dataset, batch_size=8, collate_fn=CausalLMCollator())

    # 4. Train
    train_config = TrainingConfig(
        lr=1e-3,
        total_steps=500,
        warmup_steps=50,
        log_every=50,
        checkpoint_every=500,
        checkpoint_dir="checkpoints/tiny_example",
        resume=False,
    )
    device = torch.device("cpu")
    trainer = Trainer(model, config, train_config, loader, device=device)
    trainer.train()
    print("Training complete!")


if __name__ == "__main__":
    main()
