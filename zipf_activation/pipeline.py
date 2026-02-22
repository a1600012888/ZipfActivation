from __future__ import annotations

import json
import logging
import numpy as np
import torch
from dataclasses import asdict
from pathlib import Path
from tqdm import tqdm

from .config import ZipfActivationConfig
from .data import create_dataloader
from .model import load_model, ActivationCollector
from .fitting import fit_power_law, PowerLawFitResult
from .visualization import (
    plot_rank_frequency,
    plot_multi_layer_comparison,
    plot_exponent_dashboard,
    plot_slope_dashboard,
    plot_distribution_comparison,
    plot_channel_heatmap,
)

logger = logging.getLogger(__name__)


def _ensure_dirs(base: Path) -> dict[str, Path]:
    dirs = {
        "statistics": base / "statistics",
        "fits": base / "fits",
        "plots": base / "plots",
        "plots_rank": base / "plots" / "rank_frequency",
        "plots_ccdf": base / "plots" / "ccdf",
        "plots_comparison": base / "plots" / "multi_layer",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


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
    all_stats: dict[str, dict[str, np.ndarray]] = {}
    for layer_name, acc in collector.accumulators.items():
        all_stats[layer_name] = acc.finalize()
        if config.output.save_statistics:
            out_path = dirs["statistics"] / f"{layer_name}.npz"
            np.savez_compressed(str(out_path), **all_stats[layer_name])
            logger.info("Saved statistics for %s -> %s", layer_name, out_path)

    # ---- Phase 5: Power law fitting ----
    logger.info("Fitting power laws...")
    all_fits: dict[str, dict[str, PowerLawFitResult]] = {}
    channel_fits: dict[str, list[PowerLawFitResult]] = {}

    for layer_name, stats in all_stats.items():
        all_fits[layer_name] = {}
        for metric_name, values in stats.items():
            if metric_name == "channel_wise":
                # values shape: [N, k]  – fit each channel separately
                # Use smaller subsample for channel-wise (many fits)
                from copy import copy
                ch_fit_cfg = copy(config.fitting)
                ch_fit_cfg.max_samples_for_fit = min(10_000, config.fitting.max_samples_for_fit)
                n_channels = values.shape[1] if values.ndim == 2 else 0
                ch_results = []
                for ch_idx in range(n_channels):
                    ch_vals = values[:, ch_idx]
                    fr = fit_power_law(ch_vals, f"channel_{ch_idx}", layer_name, ch_fit_cfg)
                    ch_results.append(fr)
                channel_fits[layer_name] = ch_results
            else:
                fr = fit_power_law(values, metric_name, layer_name, config.fitting)
                all_fits[layer_name][metric_name] = fr
                logger.info("  %s / %s: alpha=%.2f, slope=%.2f, R²=%.3f",
                            layer_name, metric_name, fr.alpha, fr.log_log_slope, fr.log_log_r_squared)

    # Save fit results as JSON
    fit_json = {}
    for layer_name, fits in all_fits.items():
        fit_json[layer_name] = {mn: asdict(fr) for mn, fr in fits.items()}
    json_path = dirs["fits"] / "fit_results.json"
    with open(json_path, "w") as f:
        json.dump(fit_json, f, indent=2, default=str)
    logger.info("Fit results saved to %s", json_path)

    # ---- Phase 6: Visualization ----
    logger.info("Generating plots...")

    # Per-metric, per-layer rank-frequency and CCDF plots
    for layer_name, fits in all_fits.items():
        for metric_name, fr in fits.items():
            values = all_stats[layer_name][metric_name]
            plot_rank_frequency(
                values, fr,
                dirs["plots_rank"] / f"{layer_name}_{metric_name}",
                config.output,
            )
            plot_distribution_comparison(
                values, fr,
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

    # Dashboard heatmaps
    plot_exponent_dashboard(all_fits, dirs["plots"] / "alpha_dashboard", config.output)
    plot_slope_dashboard(all_fits, dirs["plots"] / "slope_dashboard", config.output)

    # Channel-wise heatmap
    if channel_fits:
        plot_channel_heatmap(channel_fits, dirs["plots"] / "channel_heatmap", config.output)

    logger.info("All plots saved to %s", dirs["plots"])
    logger.info("Pipeline complete.")
