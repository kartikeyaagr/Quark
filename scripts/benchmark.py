"""TurboQuant benchmark suite.

Measures prefill throughput, decode throughput, and peak memory across:
  - FP32 baseline (no KV compression)
  - INT8 weight quantization
  - INT4 weight quantization
  - TurboQuant compressed KV cache (3-bit)
  - XSA variant

Run:
    uv run python scripts/benchmark.py                    # all presets on CPU
    uv run python scripts/benchmark.py --preset turbo-small --device cuda
    uv run python scripts/benchmark.py --preset turbo-tiny --decode-steps 200 --runs 5
"""

from __future__ import annotations

import argparse
import copy
import time
from dataclasses import dataclass

import torch

from turboquant.model.config import ModelConfig
from turboquant.model.transformer import Transformer
from turboquant.inference.kv_cache import build_kv_caches, build_compressed_kv_caches
from turboquant.quantization.quantize import quantize_model, model_size_mb


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sync(device: torch.device) -> None:
    """Block until all GPU kernels finish (no-op on CPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _peak_mem_mb(device: torch.device) -> float:
    """Peak allocated memory in MB (CUDA only; returns 0 elsewhere)."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024 ** 2
    return 0.0


def _reset_mem(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


# ─────────────────────────────────────────────────────────────────────────────
# Timed kernels
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def bench_prefill(
    model: Transformer,
    config: ModelConfig,
    prompt_len: int,
    batch_size: int,
    device: torch.device,
    runs: int = 3,
    warmup: int = 1,
) -> tuple[float, float]:
    """Return (mean_ms, tokens_per_sec) for a single prefill pass."""
    ids = torch.randint(0, config.vocab_size, (batch_size, prompt_len), device=device)

    for _ in range(warmup):
        _sync(device)
        model(ids)
        _sync(device)

    times = []
    for _ in range(runs):
        _sync(device)
        t0 = time.perf_counter()
        model(ids)
        _sync(device)
        times.append(time.perf_counter() - t0)

    mean_ms = (sum(times) / len(times)) * 1000
    tps = (batch_size * prompt_len) / (mean_ms / 1000)
    return mean_ms, tps


@torch.inference_mode()
def bench_decode(
    model: Transformer,
    config: ModelConfig,
    prompt_len: int,
    decode_steps: int,
    batch_size: int,
    device: torch.device,
    use_compressed_cache: bool = False,
    compressed_bits: int = 3,
    runs: int = 3,
    warmup: int = 1,
) -> tuple[float, float]:
    """Return (mean_ms_per_token, tokens_per_sec) for incremental decode."""
    ids = torch.randint(0, config.vocab_size, (batch_size, prompt_len), device=device)

    def _run_decode():
        if use_compressed_cache:
            kv = build_compressed_kv_caches(config, batch_size=batch_size,
                                             device=device, bits=compressed_bits)
        else:
            kv = build_kv_caches(config, batch_size=batch_size, device=device)

        # Prefill
        _, kv = model(ids, kv_caches=kv, cache_pos=0)
        pos = prompt_len

        # Decode loop
        token = torch.randint(0, config.vocab_size, (batch_size, 1), device=device)
        for _ in range(decode_steps):
            _, kv = model(token, kv_caches=kv, cache_pos=pos)
            pos += 1
            if pos >= config.max_seq_len:
                break

    for _ in range(warmup):
        _sync(device)
        _run_decode()
        _sync(device)

    times = []
    for _ in range(runs):
        _sync(device)
        t0 = time.perf_counter()
        _run_decode()
        _sync(device)
        times.append(time.perf_counter() - t0)

    mean_total_ms = (sum(times) / len(times)) * 1000
    actual_steps = min(decode_steps, config.max_seq_len - prompt_len)
    ms_per_tok = mean_total_ms / actual_steps
    tps = (batch_size * actual_steps) / (mean_total_ms / 1000)
    return ms_per_tok, tps


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Row:
    variant: str
    size_mb: float
    prefill_tps: float
    decode_ms: float
    decode_tps: float
    peak_mem_mb: float


def run_benchmark(
    preset: str,
    device: torch.device,
    prompt_len: int,
    decode_steps: int,
    batch_size: int,
    runs: int,
) -> None:
    print(f"\n{'='*72}")
    print(f"  Preset: {preset}  |  Device: {device}  |  "
          f"B={batch_size}  prompt={prompt_len}  decode={decode_steps} steps")
    print(f"{'='*72}")

    config = ModelConfig.from_preset(preset)
    print(f"  dim={config.dim}  layers={config.n_layers}  "
          f"heads={config.n_heads}/{config.n_kv_heads}  "
          f"max_seq={config.max_seq_len}  "
          f"~{config.n_params()/1e6:.0f}M params\n")

    variants: list[tuple[str, Transformer, bool, int]] = []

    # FP32 baseline
    m_fp32 = Transformer(config).to(device).eval()
    variants.append(("FP32", m_fp32, False, 0))

    # INT8
    m_int8 = copy.deepcopy(m_fp32)
    quantize_model(m_int8, method="int8")
    m_int8.eval()
    variants.append(("INT8", m_int8, False, 0))

    # INT4
    m_int4 = copy.deepcopy(m_fp32)
    quantize_model(m_int4, method="int4")
    m_int4.eval()
    variants.append(("INT4", m_int4, False, 0))

    # FP32 + TurboQuant compressed KV cache (3-bit)
    variants.append(("FP32 + TQ-KV (3-bit)", m_fp32, True, 3))

    # XSA variant
    config_xsa = ModelConfig(**{**config.to_dict(), "use_xsa": True})
    m_xsa = Transformer(config_xsa).to(device).eval()
    variants.append(("FP32 + XSA", m_xsa, False, 0))

    rows: list[Row] = []

    col_w = [24, 10, 16, 16, 16, 14]
    header = (f"{'Variant':<{col_w[0]}} {'Size MB':>{col_w[1]}} "
              f"{'Prefill tok/s':>{col_w[2]}} {'Decode ms/tok':>{col_w[3]}} "
              f"{'Decode tok/s':>{col_w[4]}} {'Peak VRAM MB':>{col_w[5]}}")
    sep = "-" * sum(col_w + [len(col_w) - 1])
    print(header)
    print(sep)

    for name, model, use_comp, comp_bits in variants:
        _reset_mem(device)

        size = model_size_mb(model)

        _, pre_tps = bench_prefill(
            model, config if "XSA" not in name else config_xsa,
            prompt_len, batch_size, device, runs=runs, warmup=1,
        )

        dec_ms, dec_tps = bench_decode(
            model, config if "XSA" not in name else config_xsa,
            prompt_len, decode_steps, batch_size, device,
            use_compressed_cache=use_comp,
            compressed_bits=comp_bits,
            runs=runs, warmup=1,
        )

        peak = _peak_mem_mb(device)
        rows.append(Row(name, size, pre_tps, dec_ms, dec_tps, peak))

        print(f"{name:<{col_w[0]}} {size:>{col_w[1]}.1f} "
              f"{pre_tps:>{col_w[2]}.0f} "
              f"{dec_ms:>{col_w[3]}.2f} "
              f"{dec_tps:>{col_w[4]}.0f} "
              f"{peak:>{col_w[5]}.1f}")

    # Summary ratios vs FP32
    fp32 = rows[0]
    print(f"\n  Ratios vs FP32 baseline:")
    for r in rows[1:]:
        size_ratio = fp32.size_mb / r.size_mb if r.size_mb > 0 else 1.0
        speed_ratio = r.decode_tps / fp32.decode_tps if fp32.decode_tps > 0 else 1.0
        print(f"  {r.variant:<28} size {size_ratio:4.1f}x smaller  "
              f"decode {speed_ratio:4.2f}x faster")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboQuant benchmark")
    parser.add_argument("--preset", default=None,
                        choices=["turbo-tiny", "turbo-small", "turbo-medium", "turbo-large"],
                        help="Run a single preset (default: all presets)")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cpu, cuda, mps")
    parser.add_argument("--prompt-len", type=int, default=128,
                        help="Number of prompt tokens")
    parser.add_argument("--decode-steps", type=int, default=64,
                        help="Number of autoregressive decode steps")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--runs", type=int, default=3,
                        help="Timed runs per variant (more = more stable)")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    print(f"\nTurboQuant Benchmark — device: {device}")

    presets = [args.preset] if args.preset else ["turbo-tiny", "turbo-small"]

    for preset in presets:
        # Clamp prompt+decode to fit within max_seq_len
        cfg = ModelConfig.from_preset(preset)
        prompt_len = min(args.prompt_len, cfg.max_seq_len // 2)
        decode_steps = min(args.decode_steps, cfg.max_seq_len - prompt_len - 1)

        run_benchmark(
            preset=preset,
            device=device,
            prompt_len=prompt_len,
            decode_steps=decode_steps,
            batch_size=args.batch_size,
            runs=args.runs,
        )


if __name__ == "__main__":
    main()
