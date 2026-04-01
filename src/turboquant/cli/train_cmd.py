"""CLI: turboquant train"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

train_app = typer.Typer(help="Train a TurboQuant model.")
console = Console()


@train_app.command("run")
def train_run(
    config: Path = typer.Option(..., "--config", "-c", help="Model config YAML or preset name"),
    data: list[Path] = typer.Option(..., "--data", "-d", help="Training text file(s)"),
    tokenizer: Path = typer.Option(..., "--tokenizer", "-t", help="Tokenizer directory"),
    checkpoint_dir: Path = typer.Option(Path("checkpoints"), help="Where to save checkpoints"),
    lr: float = typer.Option(3e-4, help="Peak learning rate"),
    batch_size: int = typer.Option(16, help="Batch size per device"),
    total_steps: int = typer.Option(100_000, help="Total training steps"),
    warmup_steps: int = typer.Option(2000, help="LR warmup steps"),
    grad_accum: int = typer.Option(1, help="Gradient accumulation steps"),
    grad_checkpoint: bool = typer.Option(False, help="Enable gradient checkpointing"),
    compile_model: bool = typer.Option(False, help="torch.compile model"),
    use_wandb: bool = typer.Option(False, help="Log to Weights & Biases"),
    resume: bool = typer.Option(True, help="Resume from latest checkpoint if available"),
    device: str = typer.Option("auto", help="Device: auto, cpu, cuda, mps"),
) -> None:
    import torch
    from torch.utils.data import DataLoader

    from turboquant.model.config import ModelConfig
    from turboquant.model.transformer import Transformer
    from turboquant.tokenizer.tokenizer import TurboTokenizer
    from turboquant.data.dataset import TextDataset
    from turboquant.data.collator import CausalLMCollator
    from turboquant.training.trainer import Trainer, TrainingConfig

    # Resolve device
    if device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    console.print(f"[bold green]Device:[/bold green] {dev}")

    # Load config (preset name or YAML path)
    cfg_str = str(config)
    if cfg_str.startswith("turbo-"):
        model_config = ModelConfig.from_preset(cfg_str)
    else:
        model_config = ModelConfig.from_yaml(config)

    console.print(f"[bold green]Model:[/bold green] {model_config.n_params() / 1e6:.1f}M params")

    tok = TurboTokenizer.from_file(tokenizer)
    dataset = TextDataset([str(p) for p in data], tok, seq_len=model_config.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=CausalLMCollator(),
        num_workers=2,
        pin_memory=(dev.type == "cuda"),
    )

    model = Transformer(model_config)
    train_config = TrainingConfig(
        lr=lr,
        batch_size=batch_size,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        grad_accum_steps=grad_accum,
        use_grad_checkpoint=grad_checkpoint,
        compile=compile_model,
        checkpoint_dir=str(checkpoint_dir),
        use_wandb=use_wandb,
        resume=resume,
    )

    trainer = Trainer(model, model_config, train_config, loader, device=dev)
    trainer.train()
