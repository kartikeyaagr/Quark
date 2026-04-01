"""CLI: turboquant tokenizer"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

tokenizer_app = typer.Typer(help="Train or inspect a tokenizer.")
console = Console()


@tokenizer_app.command("train")
def tokenizer_train(
    files: list[Path] = typer.Option(..., "--file", "-f", help="Text file(s) to train on"),
    output: Path = typer.Option(..., "--output", "-o", help="Directory to save the tokenizer"),
    vocab_size: int = typer.Option(32000, help="Vocabulary size"),
    min_frequency: int = typer.Option(2, help="Minimum token frequency"),
) -> None:
    from turboquant.tokenizer.trainer import BPETrainer
    trainer = BPETrainer(vocab_size=vocab_size, min_frequency=min_frequency)
    console.print(f"[bold green]Training tokenizer[/bold green] (vocab_size={vocab_size})...")
    trainer.train([str(f) for f in files], save_path=output)
    console.print(f"[bold green]Saved to[/bold green] {output}")


@tokenizer_app.command("encode")
def tokenizer_encode(
    tokenizer: Path = typer.Option(..., "--tokenizer", "-t"),
    text: str = typer.Argument(...),
) -> None:
    from turboquant.tokenizer.tokenizer import TurboTokenizer
    tok = TurboTokenizer.from_file(tokenizer)
    ids = tok.encode(text)
    console.print(f"Tokens ({len(ids)}): {ids}")


@tokenizer_app.command("info")
def tokenizer_info(
    tokenizer: Path = typer.Option(..., "--tokenizer", "-t"),
) -> None:
    from turboquant.tokenizer.tokenizer import TurboTokenizer
    tok = TurboTokenizer.from_file(tokenizer)
    console.print(f"Vocab size: {tok.vocab_size}")
    console.print(f"BOS id: {tok.bos_id}, EOS id: {tok.eos_id}, PAD id: {tok.pad_id}")
