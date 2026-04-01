"""Print parameter counts for all model presets.

Run:  uv run python scripts/count_params.py
"""

from turboquant.model.config import ModelConfig
from turboquant.model.transformer import Transformer


def main() -> None:
    print(f"\n{'Preset':<16} {'Config ~Params':>16} {'Actual Params':>16} {'Size MB':>10}")
    print("-" * 62)
    for name in ["turbo-tiny", "turbo-small", "turbo-medium", "turbo-large"]:
        cfg = ModelConfig.from_preset(name)
        m = Transformer(cfg)
        actual = m.n_params()
        config_est = cfg.n_params()
        # Estimate size assuming float32
        size_mb = actual * 4 / (1024 ** 2)
        print(f"{name:<16} {config_est / 1e6:>14.1f}M {actual / 1e6:>14.1f}M {size_mb:>9.1f}M")


if __name__ == "__main__":
    main()
