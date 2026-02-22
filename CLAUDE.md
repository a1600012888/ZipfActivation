# CLAUDE.md — Development Guide for ZipfActivation

## Quick Reference

- **Venv**: `source venv/bin/activate`
- **Run smoke test**: `python scripts/run.py --config configs/default.yaml --model.name Qwen/Qwen3-0.6B-Base --data.name wikitext --data.subset wikitext-2-raw-v1 --data.max_tokens 20000 --output.results_dir results/smoke_test --output.plot_format png`
- **Run full pipeline**: `python scripts/run.py --config configs/default.yaml`
- **Hardware**: 2x A100-80GB, 1.1TB RAM, 96 CPUs

## Architecture Overview

The pipeline flows through 6 phases, orchestrated by `pipeline.py:run_pipeline()`:

```
Load model → Register hooks → Stream data → Finalize stats → Fit power laws → Generate plots
```

**Key design principle**: hooks compute scalar reductions on GPU immediately; full activation tensors are never stored.

## Module Responsibilities

| Module | Purpose | Key classes/functions |
|--------|---------|----------------------|
| `config.py` | Dataclass configs, YAML/CLI loading | `ZipfActivationConfig`, `load_config()`, `config_from_cli()` |
| `data.py` | Streaming token-packed batches | `PackedTokenDataset`, `create_dataloader()` |
| `model.py` | Model loading, hook-based collection | `load_model()`, `ActivationCollector`, `_resolve_collection_points()` |
| `statistics.py` | 14 GPU metric functions, accumulation | `METRIC_FUNCTIONS` dict, `StatisticsAccumulator` |
| `fitting.py` | Clauset + OLS power law fitting | `fit_power_law()`, `PowerLawFitResult` |
| `visualization.py` | 6 plot types | `plot_rank_frequency()`, `plot_distribution_comparison()`, etc. |
| `pipeline.py` | End-to-end orchestration | `run_pipeline()` |

## How to Add a New Metric

1. Add the metric function to `statistics.py` following the signature pattern:
   ```python
   def metric_mymetric(flat: torch.Tensor, **kw) -> torch.Tensor:
       # flat: [N, d] float32 on GPU, return: [N] tensor
       return ...
   ```
2. Register it in the `METRIC_FUNCTIONS` dict in `statistics.py`
3. Add `"mymetric"` to the `metrics` list in `configs/default.yaml` (or `StatisticsConfig.metrics` default)
4. Everything else (accumulation, fitting, plotting) is automatic

## How to Add a New Collection Point

1. Add the symbolic name resolution in `model.py:_resolve_collection_points()`
2. Add it to `collection_points` in config

## How to Add a New Plot Type

1. Add the plotting function to `visualization.py`
2. Call it from `pipeline.py:run_pipeline()` in the Phase 6 section
3. Optionally add a new subdirectory in `_ensure_dirs()`

## How to Add a New Model Architecture

The model loading is architecture-agnostic via HuggingFace's `AutoModelForCausalLM`. Three helper functions in `model.py` handle architecture differences:
- `_get_base_model()`: tries `model.model` then `model.transformer`
- `_get_embedding_module()`: tries `embed_tokens`, `wte`, `word_embedding`, `embeddings`
- `_get_layers()`: tries `layers`, `h`, `blocks`

If a new architecture doesn't match these patterns, add the attribute name to the relevant function.

## Configuration System

`config.py` uses nested dataclasses. The loading priority is:
1. Dataclass defaults (hardcoded)
2. YAML file (if `--config` provided)
3. CLI overrides (e.g. `--model.name`)

The `_apply_dict()` function recursively applies a dict to the nested dataclass tree. To add a new CLI override, add an `argparse` argument in `config_from_cli()` and the corresponding override logic.

## Important Implementation Details

### Token Packing (`data.py`)
Documents are concatenated into a continuous token stream and chunked into `[batch_size, max_seq_len]` tensors. There is no padding — every token is real data. The `attention_mask` is all ones.

### Hook Mechanism (`model.py`)
`ActivationCollector.__init__()` registers `forward_hook` on target modules. The hook extracts hidden states (handling tuple outputs from decoder layers), reshapes to `[N_tokens, hidden_dim]`, and calls `StatisticsAccumulator.add_batch()`.

### Statistics Computation (`statistics.py`)
`StatisticsAccumulator.add_batch()` casts input to float32, computes all metrics on GPU, and appends CPU tensors to lists. `finalize()` concatenates into numpy arrays.

### Channel-wise Analysis
The `channel_wise` metric samples `channel_sample_size` (default 64) random channels using a fixed seed. Each channel gets its own power law fit. Channel-wise fits use a reduced `max_samples_for_fit` (capped at 10K in `pipeline.py`) for performance.

### Power Law Fitting (`fitting.py`)
- `powerlaw.Fit()` (Clauset method) finds optimal `xmin` and `alpha` by minimizing KS distance
- The package has a bug in v2.0.0 where `KS()` raises NameError — the code catches this and uses `scipy.stats.kstest` as fallback
- `powerlaw.Fit()` is slow: ~1 min per fit on 50K samples. The `max_samples_for_fit` config controls this tradeoff
- The Clauset alpha often saturates at ~3.0 for non-power-law distributions — the log-log slope is more informative in these cases
- Distribution comparisons use likelihood ratio tests; negative R means the alternative fits better than power law

### Visualization (`visualization.py`)
- Uses `matplotlib.use("Agg")` for headless rendering
- `_save_fig()` handles format selection (pdf/png/both)
- Plots downsample to ~10K points for rendering speed

## Output Structure

```
results/
├── statistics/
│   ├── embed.npz           # numpy arrays: rms[N], l2_norm[N], channel_wise[N,k], ...
│   ├── layer_14.npz
│   └── layer_26.npz
├── fits/
│   └── fit_results.json    # {layer: {metric: {alpha, xmin, slope, R², comparisons, ...}}}
└── plots/
    ├── rank_frequency/     # Per (layer, metric) log-log rank plots
    ├── ccdf/               # Per (layer, metric) CCDF with annotations
    ├── multi_layer/        # Per metric, all layers overlaid
    ├── alpha_dashboard.png # Heatmap: alpha by (layer, metric)
    ├── slope_dashboard.png # Heatmap: slope by (layer, metric)
    └── channel_heatmap.png # Heatmap: channel slopes by layer
```

## Known Issues and Workarounds

| Issue | Workaround | Location |
|-------|-----------|----------|
| `powerlaw` 2.0.0 `KS()` bug | try/except with scipy fallback | `fitting.py:85-93` |
| `torch_dtype` deprecated in transformers 5.2 | Use `dtype=` instead | `model.py:18` |
| `powerlaw.Fit()` very slow | Cap at 50K samples (10K for channels) | `fitting.py:78`, `pipeline.py:96` |
| wikitext dataset needs subset | Pass via `--data.subset` or config | `data.py:27` |
| Clauset alpha saturates at 3.0 | Use log-log slope + R² as complement | Interpretation note |

## Qwen3 Model Sizes (for reference)

There is no "Qwen3-3B". Available dense base models:
- `Qwen/Qwen3-0.6B-Base` (28 layers, hidden=1024) — good for smoke tests
- `Qwen/Qwen3-1.7B-Base` (28 layers, hidden=2048)
- `Qwen/Qwen3-4B-Base` (36 layers, hidden=2560) — project default
- `Qwen/Qwen3-8B-Base` (36 layers, hidden=4096)
- `Qwen/Qwen3-14B-Base`, `Qwen/Qwen3-32B-Base`

## Future Directions (Not Yet Implemented)

- **Channel dimension analysis**: Study power law across the d channels for each token position (currently only N-dimension analysis is implemented)
- **More collection points**: attention outputs, MLP intermediate activations, residual stream
- **Cross-model comparison**: Run the same analysis across different model sizes/families
