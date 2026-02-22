from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .config import OutputConfig


def _save_fig(fig: plt.Figure, path: Path, config: OutputConfig) -> None:
    formats = [config.plot_format] if config.plot_format != "both" else ["pdf", "png"]
    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"), format=fmt, dpi=config.plot_dpi,
                    bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------
# 1. Rank-frequency log-log plot (with OLS fit line)
# -----------------------------------------------------------------------
def plot_rank_frequency(
    values: np.ndarray,
    layer_name: str,
    metric_name: str,
    slope: float,
    intercept: float,
    r_squared: float,
    output_path: Path,
    config: OutputConfig,
) -> None:
    sorted_vals = np.sort(np.abs(values))[::-1]
    sorted_vals = sorted_vals[sorted_vals > 0]
    if len(sorted_vals) < 10:
        return

    ranks = np.arange(1, len(sorted_vals) + 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    step = max(1, len(ranks) // 10_000)
    ax.loglog(ranks[::step], sorted_vals[::step], ".", markersize=1, alpha=0.4, label="Data")

    # Overlay OLS fit line
    if r_squared > 0:
        fit_line = np.exp(intercept) * ranks.astype(float) ** slope
        ax.loglog(ranks[::step], fit_line[::step], "r-", linewidth=1.5,
                  label=f"OLS fit (slope={slope:.3f}, R2={r_squared:.3f})")

    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(f"{layer_name} / {metric_name}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    _save_fig(fig, output_path, config)


# -----------------------------------------------------------------------
# 2. Multi-layer comparison
# -----------------------------------------------------------------------
def plot_multi_layer_comparison(
    layer_values: dict[str, np.ndarray],
    metric_name: str,
    output_path: Path,
    config: OutputConfig,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(layer_values)))

    for (layer_name, values), color in zip(layer_values.items(), colors):
        sorted_vals = np.sort(np.abs(values))[::-1]
        sorted_vals = sorted_vals[sorted_vals > 0]
        if len(sorted_vals) < 10:
            continue
        ranks = np.arange(1, len(sorted_vals) + 1)
        step = max(1, len(ranks) // 5_000)
        ax.loglog(ranks[::step], sorted_vals[::step], ".", markersize=1.5,
                  alpha=0.6, color=color, label=layer_name)

    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(f"Multi-layer comparison: {metric_name}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    _save_fig(fig, output_path, config)


# -----------------------------------------------------------------------
# 3. Slope dashboard (heatmap)
# -----------------------------------------------------------------------
def plot_slope_dashboard(
    slopes: dict[str, dict[str, tuple[float, float, float]]],
    output_path: Path,
    config: OutputConfig,
) -> None:
    """Heatmap of log-log slopes across (layer, metric)."""
    layer_names = sorted(slopes.keys())
    metric_names: list[str] = []
    for lr in slopes.values():
        for mn in lr:
            if mn not in metric_names:
                metric_names.append(mn)

    data = np.full((len(metric_names), len(layer_names)), np.nan)
    for j, layer in enumerate(layer_names):
        for i, metric in enumerate(metric_names):
            if metric in slopes[layer]:
                data[i, j] = slopes[layer][metric][0]  # slope

    fig, ax = plt.subplots(figsize=(max(8, len(layer_names) * 2), max(6, len(metric_names) * 0.5)))
    sns.heatmap(data, annot=True, fmt=".3f", xticklabels=layer_names,
                yticklabels=metric_names, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Log-Log Slope by Layer and Metric", fontsize=14)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Metric")

    _save_fig(fig, output_path, config)


# -----------------------------------------------------------------------
# 4. CCDF plot
# -----------------------------------------------------------------------
def plot_ccdf(
    values: np.ndarray,
    layer_name: str,
    metric_name: str,
    output_path: Path,
    config: OutputConfig,
) -> None:
    """Empirical complementary CDF on log-log axes."""
    abs_values = np.sort(np.abs(values))[::-1]
    abs_values = abs_values[abs_values > 0]
    if len(abs_values) < 10:
        return

    ccdf = np.arange(1, len(abs_values) + 1) / len(abs_values)

    fig, ax = plt.subplots(figsize=(8, 6))
    step = max(1, len(abs_values) // 10_000)
    ax.loglog(abs_values[::step], ccdf[::step], ".", markersize=1, alpha=0.4, label="Empirical CCDF")

    ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("P(X >= x)", fontsize=12)
    ax.set_title(f"CCDF: {layer_name} / {metric_name}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    _save_fig(fig, output_path, config)


# -----------------------------------------------------------------------
# 5. Channel-wise rank-frequency (overlay sampled channels)
# -----------------------------------------------------------------------
def plot_channel_rank_frequency(
    channel_data: np.ndarray,
    layer_name: str,
    output_path: Path,
    config: OutputConfig,
    max_channels_to_plot: int = 16,
) -> None:
    """Overlay rank-frequency curves for a subset of sampled channels."""
    n_samples, n_channels = channel_data.shape
    # Pick evenly spaced channels to avoid clutter
    indices = np.linspace(0, n_channels - 1, min(max_channels_to_plot, n_channels), dtype=int)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab20(np.linspace(0, 1, len(indices)))

    for idx, color in zip(indices, colors):
        vals = np.sort(np.abs(channel_data[:, idx]))[::-1]
        vals = vals[vals > 0]
        if len(vals) < 10:
            continue
        ranks = np.arange(1, len(vals) + 1)
        step = max(1, len(ranks) // 3_000)
        ax.loglog(ranks[::step], vals[::step], ".", markersize=1, alpha=0.5,
                  color=color, label=f"ch {idx}")

    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(f"Channel-wise rank-frequency: {layer_name}", fontsize=14)
    ax.legend(fontsize=8, ncol=2, markerscale=5)
    ax.grid(True, alpha=0.3)

    _save_fig(fig, output_path, config)
