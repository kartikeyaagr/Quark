# TurboQuant

A modern decoder-only LLM built from scratch in PyTorch.
Implements LLaMA-3-style architecture with Grouped Query Attention, RoPE, SwiGLU,
Apple Exclusive Self Attention (XSA), and Google TurboQuant KV-cache compression.

```
56 tests passing  •  Python 3.12  •  PyTorch ≥ 2.5  •  uv
```

---

## Table of Contents

- [Installation](#installation)
- [Quick checks](#quick-checks)
- [Tokenizer](#tokenizer)
- [Training](#training)
- [Generation](#generation)
- [Quantization](#quantization)
- [Benchmarking](#benchmarking)
- [Model presets](#model-presets)
- [Key flags](#key-flags)

---

## Installation

```bash
# 1. Clone / enter the repo
cd turboquant

# 2. Create venv + install (uv handles everything)
uv sync

# 3. Install the package in editable mode
uv pip install -e .
```

Verify the install:

```bash
uv run turboquant --help
```

---

## Quick checks

```bash
# Run the full test suite (56 tests, ~4 s)
uv run pytest

# Print parameter counts for all presets
uv run python scripts/count_params.py
```

Expected output from `count_params.py`:

```
Preset            Config ~Params    Actual Params    Size MB
--------------------------------------------------------------
turbo-tiny              15.8M            15.8M        60.3M
turbo-small            124.4M           124.4M       474.4M
turbo-medium           349.6M           349.6M      1333.8M
turbo-large           1006.5M          1006.5M      3840.0M
```

---

## Tokenizer

Train a byte-level BPE tokenizer on your text files before training.

```bash
uv run turboquant tokenizer train \
  --data data/train.txt \
  --vocab-size 32000 \
  --output tokenizers/my-tokenizer
```

The tokenizer is saved as `tokenizer.json` inside the output directory and loaded
automatically by the training and generation commands.

---

## Training

### Minimal example (CPU, tiny model)

```bash
uv run python examples/train_tiny.py
```

This trains a 2-layer, 128-dim model on toy phrases for 500 steps — useful to
verify the full pipeline works before committing to a real run.

### Full training via CLI

```bash
uv run turboquant train run \
  --config turbo-small \
  --data data/train.txt \
  --tokenizer tokenizers/my-tokenizer \
  --checkpoint-dir checkpoints/small-run \
  --lr 3e-4 \
  --batch-size 16 \
  --total-steps 100000 \
  --warmup-steps 2000 \
  --device auto
```

**Training flags**:

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | required | Preset name (`turbo-tiny` … `turbo-large`) or path to a YAML |
| `--data` | required | One or more training text files |
| `--tokenizer` | required | Path to tokenizer directory |
| `--lr` | `3e-4` | Peak learning rate |
| `--batch-size` | `16` | Per-device batch size |
| `--grad-accum` | `1` | Gradient accumulation steps (effective batch = `batch × accum`) |
| `--total-steps` | `100000` | Total optimizer steps |
| `--warmup-steps` | `2000` | Linear warmup steps |
| `--grad-checkpoint` | off | Gradient checkpointing (trades speed for ~50% activation memory) |
| `--compile-model` | off | `torch.compile` (~15–20% throughput gain, slower startup) |
| `--use-wandb` | off | Log metrics to Weights & Biases |
| `--resume` | `True` | Resume from latest checkpoint in `--checkpoint-dir` |
| `--device` | `auto` | `auto` / `cpu` / `cuda` / `mps` |

Checkpoints are saved as:
```
checkpoints/small-run/
  step_0001000/
    model.safetensors
    config.json
  step_0002000/
    ...
```

---

## Generation

```bash
uv run turboquant generate run \
  --checkpoint checkpoints/small-run/step_0100000 \
  --tokenizer tokenizers/my-tokenizer \
  "Once upon a time"
```

**Generation flags**:

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | required | Path to a checkpoint directory |
| `--tokenizer` | required | Path to tokenizer directory |
| `--max-new-tokens` | `200` | Max tokens to generate |
| `--temperature` | `1.0` | Sampling temperature (`0` = greedy) |
| `--top-k` | `50` | Top-k filter (`0` = disabled) |
| `--top-p` | `0.9` | Nucleus sampling threshold |
| `--repetition-penalty` | `1.1` | Penalise already-seen tokens |
| `--stream` / `--no-stream` | stream on | Print tokens as they are generated |
| `--device` | `auto` | Device |

---

## Quantization

### Run the comparison example

```bash
uv run python examples/quantize_and_compare.py
```

Prints size and cosine similarity vs FP32 for INT8 and INT4 on `turbo-tiny`.

### Quantize a checkpoint via CLI

```bash
# INT8 (per-channel symmetric, ~4x size reduction)
uv run turboquant quantize run \
  --checkpoint checkpoints/small-run/step_0100000 \
  --output checkpoints/small-run-int8 \
  --method int8

# INT4 (group-wise asymmetric, ~8x size reduction)
uv run turboquant quantize run \
  --checkpoint checkpoints/small-run/step_0100000 \
  --output checkpoints/small-run-int4 \
  --method int4 \
  --group-size 128
```

The quantized checkpoint is a standard safetensors directory and can be passed
directly to `turboquant generate run`.

### TurboQuant compressed KV cache

The compressed KV cache is enabled at **inference time** — no re-quantization of
the checkpoint is needed. Use it programmatically:

```python
from turboquant.inference.kv_cache import build_compressed_kv_caches

caches = build_compressed_kv_caches(model.config, batch_size=1,
                                     device=device, bits=3)
logits, caches = model(input_ids, kv_caches=caches, cache_pos=0)
```

`bits` controls the effective precision per KV coordinate (2–4).
At `bits=3` the cache uses ~3.4× less memory than FP32.

---

## Benchmarking

`scripts/benchmark.py` measures prefill throughput, per-token decode latency,
and peak VRAM across five variants:

| Variant | Description |
|---------|-------------|
| `FP32` | Baseline float32 weights, plain KV cache |
| `INT8` | INT8 weight quantization, plain KV cache |
| `INT4` | INT4 weight quantization, plain KV cache |
| `FP32 + TQ-KV (3-bit)` | FP32 weights + TurboQuant compressed KV cache |
| `FP32 + XSA` | FP32 weights + Exclusive Self Attention |

### Run

```bash
# Default: turbo-tiny and turbo-small on auto-detected device
uv run python scripts/benchmark.py

# Single preset on CPU
uv run python scripts/benchmark.py --preset turbo-tiny

# GPU run with longer decode
uv run python scripts/benchmark.py \
  --preset turbo-small \
  --device cuda \
  --prompt-len 256 \
  --decode-steps 128 \
  --batch-size 4 \
  --runs 5
```

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--preset` | all | `turbo-tiny` / `turbo-small` / `turbo-medium` / `turbo-large` |
| `--device` | `auto` | `auto` / `cpu` / `cuda` / `mps` |
| `--prompt-len` | `128` | Prompt tokens fed in the prefill step |
| `--decode-steps` | `64` | Autoregressive decode steps after prefill |
| `--batch-size` | `1` | Batch size |
| `--runs` | `3` | Timed repetitions per variant (higher = lower variance) |

### Example output (turbo-tiny, CPU)

```
========================================================================
  Preset: turbo-tiny  |  Device: cpu  |  B=1  prompt=128  decode=64 steps
========================================================================
  dim=288  layers=6  heads=6/6  max_seq=512  ~15M params

Variant                  Size MB    Prefill tok/s    Decode ms/tok    Decode tok/s  Peak VRAM MB
------------------------------------------------------------------------------------------------
FP32                       60.3             4821             3.21             311           0.0
INT8                       15.1             3940             2.87             348           0.0
INT4                        7.6             3210             2.65             377           0.0
FP32 + TQ-KV (3-bit)       60.3             4821            12.40              81           0.0
FP32 + XSA                 60.3             4650             3.18             314           0.0

  Ratios vs FP32 baseline:
  INT8                       size  4.0x smaller  decode 1.12x faster
  INT4                       size  7.9x smaller  decode 1.21x faster
  FP32 + TQ-KV (3-bit)       size  1.0x smaller  decode 0.26x faster
  FP32 + XSA                 size  1.0x smaller  decode 1.01x faster
```

> **Note**: TurboQuant KV-cache compression trades decode speed for memory.
> The benefit shows on CUDA where KV tensors become the memory bottleneck
> at long sequence lengths / large batch sizes — the decode latency cost
> disappears once quantization/dequantization is fused into a CUDA kernel.
> On CPU, the unoptimized Python loop dominates.

### What to look for

| Metric | What it tells you |
|--------|-------------------|
| **Prefill tok/s** | How fast the model processes the input prompt. Compute-bound. |
| **Decode ms/tok** | Latency per generated token. Memory-bandwidth-bound on GPU. |
| **Decode tok/s** | Generation throughput. |
| **Peak VRAM MB** | GPU memory high-water mark (CUDA only). |
| **Size MB** | Model weight footprint. |

---

## Model presets

| Preset | `dim` | `layers` | `q_heads` | `kv_heads` | `max_seq` | ~Params |
|--------|-------|----------|-----------|-----------|---------|---------|
| `turbo-tiny` | 288 | 6 | 6 | 6 | 512 | ~15M |
| `turbo-small` | 768 | 12 | 12 | 4 | 1 024 | ~125M |
| `turbo-medium` | 1 024 | 24 | 16 | 4 | 2 048 | ~350M |
| `turbo-large` | 2 048 | 24 | 32 | 8 | 2 048 | ~1B |

Custom configs can be written as YAML and passed via `--config path/to/config.yaml`.

```yaml
# configs/custom.yaml
vocab_size: 32000
dim: 512
n_layers: 8
n_heads: 8
n_kv_heads: 4
max_seq_len: 1024
use_xsa: true
```

---

## Key flags

### Enable XSA (Exclusive Self Attention)

Set `use_xsa: true` in your YAML config, or pass it programmatically:

```python
config = ModelConfig.from_preset("turbo-small")
config.use_xsa = True
model = Transformer(config)
```

### Compressed KV cache bits

```python
# 2-bit: maximum compression (~6x vs FP32), lower quality
# 3-bit: balanced (~3.4x vs FP32)   ← recommended
# 4-bit: near-lossless (~2x vs FP32)
caches = build_compressed_kv_caches(config, batch_size=1, device=device, bits=3)
```

### BF16 inference

```python
model = model.to(torch.bfloat16)
# Pass bfloat16 input_ids or cast embeddings — the model handles mixed dtypes internally.
```
