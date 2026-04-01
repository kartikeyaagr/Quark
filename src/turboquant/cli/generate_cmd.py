"""CLI: turboquant generate"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

generate_app = typer.Typer(help="Generate text from a trained model.")
console = Console()


@generate_app.command("run")
def generate_run(
    checkpoint: Path = typer.Option(..., "--checkpoint", "-c", help="Checkpoint directory"),
    tokenizer: Path = typer.Option(..., "--tokenizer", "-t", help="Tokenizer directory"),
    prompt: str = typer.Argument(..., help="Prompt text"),
    max_new_tokens: int = typer.Option(200, help="Maximum new tokens to generate"),
    temperature: float = typer.Option(1.0, help="Sampling temperature (0=greedy)"),
    top_k: int = typer.Option(50, help="Top-k sampling (0=disabled)"),
    top_p: float = typer.Option(0.9, help="Nucleus sampling threshold"),
    repetition_penalty: float = typer.Option(1.1, help="Repetition penalty"),
    stream: bool = typer.Option(True, help="Stream output token by token"),
    device: str = typer.Option("auto", help="Device: auto, cpu, cuda, mps"),
) -> None:
    import torch

    from turboquant.model.config import ModelConfig
    from turboquant.model.transformer import Transformer
    from turboquant.tokenizer.tokenizer import TurboTokenizer
    from turboquant.training.checkpointing import load_checkpoint, load_config
    from turboquant.inference.generator import Generator

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else
                           "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        dev = torch.device(device)

    model_config = load_config(checkpoint)
    model = Transformer(model_config)
    load_checkpoint(checkpoint, model, device=dev)
    model = model.to(dev)

    tok = TurboTokenizer.from_file(tokenizer)
    gen = Generator(model, model_config, tok, device=dev)

    console.print(f"\n[bold cyan]Prompt:[/bold cyan] {prompt}\n")
    console.print("[bold cyan]Generated:[/bold cyan]", end=" ")

    if stream:
        for token_text in gen.stream(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ):
            console.print(token_text, end="", highlight=False)
        console.print()
    else:
        output = gen.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        console.print(output)
