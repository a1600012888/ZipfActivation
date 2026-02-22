# ZipfActivation

Research toolkit for measuring hidden activations of pretrained LLMs and testing whether they follow Zipf / power law distributions.

Given a pretrained language model and a text dataset, this pipeline:

1. Inserts hook-based feature collectors at configurable layers (embedding, middle, pre-last, or any explicit layer index)
2. Streams tokens through the model, computing scalar statistics from each token's d-dimensional activation vector on-the-fly (never storing full activations)
3. Fits power law distributions to the resulting N-length statistic arrays using both the Clauset et al. method and log-log linear regression
4. Generates diagnostic plots: rank-frequency, CCDF, multi-layer comparisons, and heatmap dashboards

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
│   ├── statistics.py           # 14 metric functions + StatisticsAccumulator
│   ├── fitting.py              # Power law fitting (powerlaw pkg + log-log OLS)
│   ├── visualization.py        # 6 plot types (rank-freq, CCDF, comparison, dashboards, heatmap)
│   └── pipeline.py             # End-to-end orchestration
├── requirements.txt
└── results/                    # Created at runtime (gitignored)
    ├── statistics/             # Per-layer .npz files with all metric arrays
    ├── fits/                   # fit_results.json with all power law fit parameters
    └── plots/                  # All generated plots
        ├── rank_frequency/
        ├── ccdf/
        └── multi_layer/
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

This runs Qwen3-4B-Base on OpenWebText (10M tokens) with 3 collection points and all 14 metrics.

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
- `"middle"` — layer at `num_layers // 2`
- `"pre_last"` — second-to-last layer
- `"layer_N"` — explicit layer index N

Or set `explicit_layers: [0, 5, 10, 15, 20, 25]` for arbitrary layer indices.

### Metrics (14 total)
Each metric reduces a d-dimensional activation vector to a scalar, yielding an N-length distribution across tokens:

| Metric | Description |
|--------|-------------|
| `channel_wise` | Raw values for k sampled channels (fit per-channel) |
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

### Fitting
- **Clauset method** (via `powerlaw` package): estimates alpha, xmin via KS statistic minimization
- **Log-log OLS**: fast linear regression on log(rank) vs log(value)
- **Distribution comparisons**: power law vs lognormal, exponential, truncated power law (likelihood ratio tests)
- Subsampled to `max_samples_for_fit` (default 50K) for performance

## Output

### Statistics (`results/statistics/`)
Per-layer `.npz` files containing numpy arrays for each metric. Load with:
```python
data = np.load("results/statistics/layer_14.npz")
rms_values = data["rms"]  # shape: [N]
channel_values = data["channel_wise"]  # shape: [N, k]
```

### Fit Results (`results/fits/fit_results.json`)
JSON with per-layer, per-metric fit parameters: alpha, xmin, sigma, KS statistic, distribution comparison results, log-log slope/intercept/R^2.

### Plots (`results/plots/`)
- `rank_frequency/` — log-log rank vs value with fit line overlay (per layer, per metric)
- `ccdf/` — complementary CDF with all fit statistics annotated
- `multi_layer/` — overlay of all layers for each metric
- `alpha_dashboard` — heatmap of power law exponents across layers and metrics
- `slope_dashboard` — heatmap of log-log slopes
- `channel_heatmap` — heatmap of per-channel slopes across layers

## Memory Efficiency

The pipeline never stores full activation tensors. Each forward hook computes scalar reductions on GPU immediately and moves only the reduced values to CPU. For 10M tokens with 14 metrics, total CPU accumulator memory is ~560 MB.

## Known Issues

- `powerlaw` 2.0.0 has a bug in `KS()` (NameError on `compute_distance_metrics`). The code includes a workaround using scipy's KS test as fallback.
- `powerlaw.Fit()` is slow (~1 minute per fit on 50K samples). Channel-wise fits use a reduced sample size (10K) to keep total time manageable.
- The Clauset method often saturates at alpha=3.0 for concentrated distributions — check the log-log slope and R^2 for a complementary view.
