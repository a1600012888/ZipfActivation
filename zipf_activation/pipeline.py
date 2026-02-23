from __future__ import annotations

import logging
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from .config import ZipfActivationConfig
from .data import create_dataloader
from .model import load_model, ActivationCollector
from .visualization import (
    TRIM_LEVELS,
    _trim_percentile,
    plot_rank_frequency,
    plot_histogram,
    plot_multi_layer_comparison,
    plot_channel_rank_frequency,
    plot_channel_histogram,
)

logger = logging.getLogger(__name__)


def _ensure_dirs(base: Path) -> dict[str, Path]:
    dirs = {
        "plots": base / "plots",
        "plots_rank": base / "plots" / "rank_frequency",
        "plots_hist": base / "plots" / "histogram",
        "plots_comparison": base / "plots" / "multi_layer",
        "plots_channel_rank": base / "plots" / "channel_rank_frequency",
        "plots_channel_hist": base / "plots" / "channel_histogram",
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

    # ---- Phase 2: Register hooks (always streaming) ----
    collector = ActivationCollector(model, config.collection, config.statistics)
    logger.info("Hooks registered on: %s", list(collector.accumulators.keys()))

    # ---- Phase 3: Stream data and collect statistics ----
    logger.info("Starting data collection (max_tokens=%d, reservoir=%d)",
                config.data.max_tokens, config.collection.reservoir_size)
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

    # ---- Phase 5: Visualization ----
    logger.info("Generating plots...")

    scalar_metrics = [m for m in config.statistics.metrics if m != "channel_wise"]

    # Per (layer, scalar_metric): rank-frequency + histogram × 4 trim levels
    for layer_name, stats in all_stats.items():
        for metric_name in scalar_metrics:
            if metric_name not in stats:
                continue
            raw_values = stats[metric_name]
            for trim_pct, trim_label in TRIM_LEVELS:
                trimmed = _trim_percentile(raw_values, trim_pct)
                plot_rank_frequency(
                    trimmed, layer_name, metric_name,
                    dirs["plots_rank"] / f"{layer_name}_{metric_name}_{trim_label}",
                    config.output, trim_pct=trim_pct, trim_label=trim_label,
                )
                plot_histogram(
                    trimmed, layer_name, metric_name,
                    dirs["plots_hist"] / f"{layer_name}_{metric_name}_{trim_label}",
                    config.output, trim_pct=trim_pct, trim_label=trim_label,
                )

    # Multi-layer comparison plots × 4 trim levels
    for metric_name in scalar_metrics:
        layer_values = {}
        for layer_name in all_stats:
            if metric_name in all_stats[layer_name]:
                layer_values[layer_name] = all_stats[layer_name][metric_name]
        if layer_values:
            for trim_pct, trim_label in TRIM_LEVELS:
                plot_multi_layer_comparison(
                    layer_values, metric_name,
                    dirs["plots_comparison"] / f"comparison_{metric_name}_{trim_label}",
                    config.output, trim_pct=trim_pct, trim_label=trim_label,
                )

    # Channel-wise plots × 4 trim levels (rank-frequency + histogram)
    for layer_name, stats in all_stats.items():
        if "channel_wise" not in stats:
            continue
        ch_data = stats["channel_wise"]  # [N, k]
        if ch_data.ndim != 2 or ch_data.shape[1] == 0:
            continue
        for trim_pct, trim_label in TRIM_LEVELS:
            plot_channel_rank_frequency(
                ch_data, layer_name,
                dirs["plots_channel_rank"] / f"{layer_name}_channels_{trim_label}",
                config.output, trim_pct=trim_pct, trim_label=trim_label,
            )
            plot_channel_histogram(
                ch_data, layer_name,
                dirs["plots_channel_hist"] / f"{layer_name}_channels_{trim_label}",
                config.output, trim_pct=trim_pct, trim_label=trim_label,
            )

    logger.info("All plots saved to %s", dirs["plots"])
    logger.info("Pipeline complete.")
