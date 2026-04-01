"""CLI: turboquant quantize"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

quantize_app = typer.Typer(help="Quantize a trained model.")
console = Console()


@quantize_app.command("run")
def quantize_run(
    checkpoint: Path = typer.Option(..., "--checkpoint", "-c"),
    output: Path = typer.Option(..., "--output", "-o", help="Output checkpoint directory"),
    method: str = typer.Option("int8", help="Quantization method: int8, int4, dynamic"),
    group_size: int = typer.Option(128, help="Group size for INT4 quantization"),
    skip_lm_head: bool = typer.Option(True, help="Skip quantizing the LM head"),
    device: str = typer.Option("cpu", help="Device to load model on"),
) -> None:
    import torch
    from turboquant.model.config import ModelConfig
    from turboquant.model.transformer import Transformer
    from turboquant.training.checkpointing import load_checkpoint, load_config, save_checkpoint
    from turboquant.quantization.quantize import quantize_model, model_size_mb

    dev = torch.device(device)
    model_config = load_config(checkpoint)
    model = Transformer(model_config)
    load_checkpoint(checkpoint, model, device=dev)
    model = model.to(dev)

    original_mb = model_size_mb(model)
    console.print(f"[bold]Original size:[/bold] {original_mb:.1f} MB")

    skip = ["lm_head"] if skip_lm_head else []
    model = quantize_model(model, method=method, int4_group_size=group_size, skip_modules=skip)  # type: ignore[arg-type]

    quantized_mb = model_size_mb(model)
    ratio = original_mb / quantized_mb

    table = Table(title="Quantization Results")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Method", method)
    table.add_row("Original size", f"{original_mb:.1f} MB")
    table.add_row("Quantized size", f"{quantized_mb:.1f} MB")
    table.add_row("Compression ratio", f"{ratio:.1f}x")
    console.print(table)

    # Save as a new checkpoint (step=0 to indicate it's a quantized artifact)
    save_checkpoint(model, model_config, step=0, checkpoint_dir=output)
    console.print(f"[bold green]Quantized model saved to:[/bold green] {output}")
