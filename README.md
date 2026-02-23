# ZipfActivation

Research toolkit for measuring hidden activations of pretrained LLMs and testing whether they follow Zipf / power law distributions.

Given a pretrained language model and a text dataset, this pipeline:

1. Inserts hook-based feature collectors at configurable layers (embedding, first, middle, last, or any explicit layer index)
2. Streams tokens through the model, computing scalar statistics from each token's d-dimensional activation vector on-the-fly using reservoir sampling (never storing full activations)
3. Generates diagnostic plots at multiple trim levels: rank-frequency, histograms, multi-layer comparisons, and channel-wise overlays

## Project Structure

```
ZipfActivation/
├── configs/
│   └── default.yaml            # Default configuration (Qwen3-4B-Base, OpenWebText, 10M tokens)
├── scripts/
│   └── run.py                  # CLI entry point
├── zipf_activation/
│   ├── __init__.py
│   ├── config.py               # Dataclass configs + YAML/CLI loading
│   ├── data.py                 # Streaming dataset with token packing (no padding waste)
│   ├── model.py                # Model loading, hook registration, ActivationCollector
│   ├── statistics.py           # 14 metric functions + StreamingStatisticsAccumulator
│   ├── visualization.py        # Plot types: rank-frequency, histogram, multi-layer, channel overlays
│   └── pipeline.py             # End-to-end orchestration
├── requirements.txt
└── results/                    # Created at runtime (gitignored)
    └── plots/
        ├── rank_frequency/
        ├── histogram/
        ├── multi_layer/
        ├── channel_rank_frequency/
        └── channel_histogram/
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+ and a CUDA-capable GPU.

## Usage

### Full run with default config

```bash
python scripts/run.py --config configs/default.yaml
```

This runs Qwen3-4B-Base on OpenWebText (10M tokens) with 4 collection points (embed, first, middle, last) and all 14 metrics.

### Quick smoke test

```bash
python scripts/run.py \
    --config configs/default.yaml \
    --model.name Qwen/Qwen3-0.6B-Base \
    --data.name wikitext \
    --data.subset wikitext-2-raw-v1 \
    --data.max_tokens 20000 \
    --output.results_dir results/smoke_test \
    --output.plot_format png
```

### Adding explicit layers

Use `--layers` to add specific layer indices on top of the default collection points:

```bash
python scripts/run.py \
    --config configs/default.yaml \
    --layers 5,10,20
```

This collects from embed + first + middle + last + layers 5, 10, and 20.

### CLI options

| Flag | Description |
|------|-------------|
| `--config` | Path to YAML config file |
| `--model.name` | HuggingFace model ID (e.g. `Qwen/Qwen3-4B-Base`) |
| `--data.name` | HuggingFace dataset ID (e.g. `Skylion007/openwebtext`) |
| `--data.subset` | Dataset config/subset name (e.g. `wikitext-2-raw-v1`) |
| `--data.max_tokens` | Number of tokens to process |
| `--data.batch_size` | Batch size for inference |
| `--data.max_seq_len` | Sequence length for token packing |
| `--output.results_dir` | Output directory for results |
| `--output.plot_format` | Plot format: `pdf`, `png`, or `both` |
| `--layers` | Comma-separated layer indices to add (e.g. `5,10,20`) |

CLI flags override the YAML config values.

## Configuration

All settings are in `configs/default.yaml`. Key sections:

### Model
- Any HuggingFace `AutoModelForCausalLM`-compatible model
- Tested with Qwen3 family (0.6B, 4B), supports LLaMA, GPT-2, Mistral, etc.
- Runs in bfloat16 with `device_map="auto"` for multi-GPU

### Data
- Any HuggingFace text dataset with a `"text"` field
- Token packing: documents are concatenated and chunked into fixed `[batch_size, max_seq_len]` sequences (no padding)
- Streaming mode avoids downloading entire datasets

### Collection Points
Symbolic names resolved at runtime:
- `"embed"` — after the embedding layer
- `"first"` — first decoder layer (layer 0)
- `"middle"` — layer at `num_layers // 2`
- `"last"` — final decoder layer
- `"layer_N"` — explicit layer index N

The `--layers` CLI flag adds extra layers on top of the symbolic defaults (it does not replace them).

### Metrics (14 total)
Each metric reduces a d-dimensional activation vector to a scalar, yielding an N-length distribution across tokens:

| Metric | Description |
|--------|-------------|
| `channel_wise` | Raw values for k sampled channels (plotted per-channel) |
| `rms` | Root mean square |
| `l2_norm` | L2 (Euclidean) norm |
| `l1_norm` | L1 (Manhattan) norm |
| `linf_norm` | L-infinity norm (max absolute value) |
| `variance` | Variance across dimensions |
| `kurtosis` | Excess kurtosis (m4/m2^2 - 3) |
| `skewness` | Skewness (m3/m2^1.5) |
| `entropy` | Softmax entropy |
| `gini` | Gini coefficient of absolute values |
| `participation_ratio` | (sum\|x\|)^2 / sum(x^2) |
| `mean_abs` | Mean absolute value |
| `max_value` | Maximum value |
| `hoyer` | Hoyer sparsity measure |

## Output

All output goes to `results/plots/` (configurable via `--output.results_dir`). Every plot type is generated at four trim levels:

| Trim Level | Meaning |
|------------|---------|
| `full` | All data points |
| `trim1` | Remove bottom 1% and top 1% (keep middle 98%) |
| `trim5` | Remove bottom 5% and top 5% (keep middle 90%) |
| `trim10` | Remove bottom 10% and top 10% (keep middle 80%) |

### Plot types

```
results/plots/
├── rank_frequency/               # Log-log rank vs value
│   ├── {layer}_{metric}_full.png
│   ├── {layer}_{metric}_trim1.png
│   ├── {layer}_{metric}_trim5.png
│   └── {layer}_{metric}_trim10.png
├── histogram/                    # Value distribution histogram
│   ├── {layer}_{metric}_full.png
│   ├── ...
├── multi_layer/                  # All layers overlaid per metric
│   ├── comparison_{metric}_full.png
│   ├── ...
├── channel_rank_frequency/       # Sampled channels overlaid per layer
│   ├── {layer}_channels_full.png
│   ├── ...
└── channel_histogram/            # Channel histograms overlaid per layer
    ├── {layer}_channels_full.png
    └── ...
```

With default settings (4 layers, 13 scalar metrics, 4 trims), this produces:
- 208 rank-frequency plots
- 208 histogram plots
- 52 multi-layer comparison plots
- 16 channel rank-frequency plots
- 16 channel histogram plots
- **500 plots total**

## Memory Efficiency

The pipeline never stores full activation tensors. Each forward hook computes scalar reductions on GPU immediately and uses reservoir sampling (Algorithm R) to maintain a bounded-size sample per metric. Memory usage is fixed regardless of how many tokens are processed — controlled by `reservoir_size` (default 1M samples per metric per layer).

## Known Issues

- `torch_dtype` is deprecated in transformers 5.2 — the code uses `dtype=` instead.
- `wikitext` dataset requires the `subset` parameter (e.g. `--data.subset wikitext-2-raw-v1`).
- There is no "Qwen3-3B". Available Qwen3 base models: 0.6B, 1.7B, 4B, 8B, 14B, 32B.
