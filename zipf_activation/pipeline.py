from __future__ import annotations

import json
import logging
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from .config import ZipfActivationConfig
from .data import create_dataloader
from .model import load_model, ActivationCollector
from .visualization import (
    plot_rank_frequency,
    plot_multi_layer_comparison,
    plot_slope_dashboard,
    plot_ccdf,
    plot_channel_rank_frequency,
)

logger = logging.getLogger(__name__)


def _ensure_dirs(base: Path) -> dict[str, Path]:
    dirs = {
        "statistics": base / "statistics",
        "plots": base / "plots",
        "plots_rank": base / "plots" / "rank_frequency",
        "plots_ccdf": base / "plots" / "ccdf",
        "plots_comparison": base / "plots" / "multi_layer",
        "plots_channel": base / "plots" / "channel_rank_frequency",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _fit_log_log(values: np.ndarray) -> tuple[float, float, float]:
    """Fast OLS on log(rank) vs log(value). Returns (slope, intercept, R^2)."""
    sorted_vals = np.sort(np.abs(values))[::-1]
    sorted_vals = sorted_vals[sorted_vals > 0]
    if len(sorted_vals) < 10:
        return 0.0, 0.0, 0.0
    ranks = np.arange(1, len(sorted_vals) + 1, dtype=np.float64)
    log_ranks = np.log(ranks)
    log_vals = np.log(sorted_vals)
    slope, intercept = np.polyfit(log_ranks, log_vals, 1)
    fitted = slope * log_ranks + intercept
    ss_res = np.sum((log_vals - fitted) ** 2)
    ss_tot = np.sum((log_vals - np.mean(log_vals)) ** 2)
    r_squared = 1.0 - ss_res / (ss_tot + 1e-12)
    return float(slope), float(intercept), float(r_squared)


def run_pipeline(config: ZipfActivationConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    results_dir = Path(config.output.results_dir)
    dirs = _ensure_dirs(results_dir)

    # ---- Phase 1: Load model & tokenizer ----
    logger.info("Loading model: %s", config.model.name)
    model, tokenizer = load_model(config.model)
    logger.info("Model loaded. Hidden size: %d, Layers: %d",
                model.config.hidden_size,
                getattr(model.config, "num_hidden_layers", "?"))

    # ---- Phase 2: Register hooks ----
    collector = ActivationCollector(model, config.collection, config.statistics)
    logger.info("Hooks registered on: %s", list(collector.accumulators.keys()))

    # ---- Phase 3: Stream data and collect statistics ----
    logger.info("Starting data collection (max_tokens=%d)", config.data.max_tokens)
    dataloader = create_dataloader(config.data, tokenizer)

    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting activations"):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            model(input_ids=input_ids, attention_mask=attention_mask)

            total_tokens += int(attention_mask.sum().item())

    collector.remove_hooks()
    logger.info("Collection complete. Total tokens: %d", total_tokens)

    # ---- Phase 4: Finalize statistics ----
    logger.info("Finalizing statistics...")
    all_stats: dict[str, dict[str, np.ndarray]] = {}
    for layer_name, acc in collector.accumulators.items():
        all_stats[layer_name] = acc.finalize()
        if config.output.save_statistics:
            out_path = dirs["statistics"] / f"{layer_name}.npz"
            np.savez_compressed(str(out_path), **all_stats[layer_name])
            logger.info("Saved statistics for %s -> %s", layer_name, out_path)

    # ---- Phase 5: Log-log regression (fast) ----
    logger.info("Computing log-log slopes...")
    # slopes[layer][metric] = (slope, intercept, r_squared)
    slopes: dict[str, dict[str, tuple[float, float, float]]] = {}
    for layer_name, stats in all_stats.items():
        slopes[layer_name] = {}
        for metric_name, values in stats.items():
            if metric_name == "channel_wise":
                continue
            s, i, r2 = _fit_log_log(values)
            slopes[layer_name][metric_name] = (s, i, r2)
            logger.info("  %s / %s: slope=%.3f, R2=%.3f", layer_name, metric_name, s, r2)

    # Save slopes as JSON
    slopes_json = {
        ln: {mn: {"slope": s, "intercept": i, "r_squared": r2}
             for mn, (s, i, r2) in metrics.items()}
        for ln, metrics in slopes.items()
    }
    slopes_path = results_dir / "slopes.json"
    with open(slopes_path, "w") as f:
        json.dump(slopes_json, f, indent=2)
    logger.info("Slopes saved to %s", slopes_path)

    # ---- Phase 6: Visualization ----
    logger.info("Generating plots...")

    # Per-metric, per-layer rank-frequency and CCDF plots
    for layer_name, layer_slopes in slopes.items():
        for metric_name, (slope, intercept, r2) in layer_slopes.items():
            values = all_stats[layer_name][metric_name]
            plot_rank_frequency(
                values, layer_name, metric_name, slope, intercept, r2,
                dirs["plots_rank"] / f"{layer_name}_{metric_name}",
                config.output,
            )
            plot_ccdf(
                values, layer_name, metric_name,
                dirs["plots_ccdf"] / f"{layer_name}_{metric_name}",
                config.output,
            )

    # Multi-layer comparison plots (one per metric)
    scalar_metrics = [m for m in config.statistics.metrics if m != "channel_wise"]
    for metric_name in scalar_metrics:
        layer_values = {}
        for layer_name in all_stats:
            if metric_name in all_stats[layer_name]:
                layer_values[layer_name] = all_stats[layer_name][metric_name]
        if layer_values:
            plot_multi_layer_comparison(
                layer_values, metric_name,
                dirs["plots_comparison"] / f"comparison_{metric_name}",
                config.output,
            )

    # Slope dashboard heatmap
    plot_slope_dashboard(slopes, dirs["plots"] / "slope_dashboard", config.output)

    # Channel-wise rank-frequency plots (one per layer, overlay all channels)
    for layer_name, stats in all_stats.items():
        if "channel_wise" in stats:
            ch_data = stats["channel_wise"]  # [N, k]
            if ch_data.ndim == 2 and ch_data.shape[1] > 0:
                plot_channel_rank_frequency(
                    ch_data, layer_name,
                    dirs["plots_channel"] / f"{layer_name}_channels",
                    config.output,
                )

    logger.info("All plots saved to %s", dirs["plots"])
    logger.info("Pipeline complete.")
