"""TurboQuant CLI entry point."""

import typer

app = typer.Typer(
    name="turboquant",
    help="TurboQuant — LLM from scratch with modern architecture and quantization.",
    add_completion=False,
)


def _register_commands() -> None:
    from turboquant.cli.train_cmd import train_app
    from turboquant.cli.generate_cmd import generate_app
    from turboquant.cli.tokenizer_cmd import tokenizer_app
    from turboquant.cli.quantize_cmd import quantize_app

    app.add_typer(train_app, name="train")
    app.add_typer(generate_app, name="generate")
    app.add_typer(tokenizer_app, name="tokenizer")
    app.add_typer(quantize_app, name="quantize")


_register_commands()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
