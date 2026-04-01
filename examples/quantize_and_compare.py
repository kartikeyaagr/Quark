"""Compare float vs INT8 vs INT4 model size and output quality.

Run:  uv run python examples/quantize_and_compare.py
"""

import torch
import torch.nn.functional as F
from turboquant.model.config import ModelConfig
from turboquant.model.transformer import Transformer
from turboquant.quantization.quantize import quantize_model, model_size_mb
import copy


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.view(1, -1), b.view(1, -1)).item()


def main() -> None:
    config = ModelConfig.from_preset("turbo-tiny")
    model_fp = Transformer(config)
    model_fp.eval()

    ids = torch.randint(0, config.vocab_size, (1, 32))

    with torch.no_grad():
        logits_fp, _ = model_fp(ids)

    results = [("float32", model_fp, logits_fp)]

    for method in ("int8", "int4"):
        m = copy.deepcopy(model_fp)
        quantize_model(m, method=method)  # type: ignore[arg-type]
        m.eval()
        with torch.no_grad():
            logits_q, _ = m(ids)
        results.append((method, m, logits_q))

    print(f"\n{'Method':<12} {'Size (MB)':<12} {'Cosine Sim':<12}")
    print("-" * 36)
    for name, m, logits in results:
        size = model_size_mb(m)
        sim = cosine_sim(logits_fp, logits)
        print(f"{name:<12} {size:<12.1f} {sim:<12.4f}")


if __name__ == "__main__":
    main()
