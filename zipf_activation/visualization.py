from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from .config import OutputConfig


def _save_fig(fig: plt.Figure, path: Path, config: OutputConfig) -> None:
    formats = [config.plot_format] if config.plot_format != "both" else ["pdf", "png"]
    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"), format=fmt, dpi=config.plot_dpi,
                    bbox_inches="tight")
    plt.close(fig)


def _trim_percentile(values: np.ndarray, pct: float) -> np.ndarray:
    """Remove bottom pct% and top pct% of values (symmetric trim).

    pct=0 returns all values. pct=1 removes bottom 1% and top 1% (keeps middle 98%).
    """
    if pct <= 0 or len(values) == 0:
        return values
    lo = np.percentile(values, pct)
    hi = np.percentile(values, 100.0 - pct)
    return values[(values >= lo) & (values <= hi)]


TRIM_LEVELS = [
    (0, "full"),
    (1, "trim1"),
    (5, "trim5"),
    (10, "trim10"),
]


# -----------------------------------------------------------------------
# 1. Rank-frequency log-log plot (no fit line)
# -----------------------------------------------------------------------
def plot_rank_frequency(
    values: np.ndarray,
    layer_name: str,
    metric_name: str,
    output_path: Path,
    config: OutputConfig,
    trim_pct: float = 0,
    trim_label: str = "full",
) -> None:
    sorted_vals = np.sort(np.abs(values))[::-1]
    sorted_vals = sorted_vals[sorted_vals > 0]
    if len(sorted_vals) < 10:
        return

    ranks = np.arange(1, len(sorted_vals) + 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    step = max(1, len(ranks) // 10_000)
    ax.loglog(ranks[::step], sorted_vals[::step], ".", markersize=1, alpha=0.4, label="Data")

    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    title = f"{layer_name} / {metric_name}"
    if trim_pct > 0:
        title += f" ({trim_label}: middle {100 - 2*trim_pct:.0f}%)"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    _save_fig(fig, output_path, config)


# -----------------------------------------------------------------------
# 2. Histogram plot
# -----------------------------------------------------------------------
def plot_histogram(
    values: np.ndarray,
    layer_name: str,
    metric_name: str,
    output_path: Path,
    config: OutputConfig,
    trim_pct: float = 0,
    trim_label: str = "full",
    n_bins: int = 100,
) -> None:
    vals = values[np.isfinite(values)]
    if len(vals) < 10:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(vals, bins=n_bins, density=True, alpha=0.7, edgecolor="black", linewidth=0.3)

    ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    title = f"{layer_name} / {metric_name}"
    if trim_pct > 0:
        title += f" ({trim_label}: middle {100 - 2*trim_pct:.0f}%)"
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    _save_fig(fig, output_path, config)


# -----------------------------------------------------------------------
# 3. Multi-layer comparison (rank-frequency, with trim)
# -----------------------------------------------------------------------
def plot_multi_layer_comparison(
    layer_values: dict[str, np.ndarray],
    metric_name: str,
    output_path: Path,
    config: OutputConfig,
    trim_pct: float = 0,
    trim_label: str = "full",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(layer_values)))

    for (layer_name, values), color in zip(layer_values.items(), colors):
        trimmed = _trim_percentile(values, trim_pct)
        sorted_vals = np.sort(np.abs(trimmed))[::-1]
        sorted_vals = sorted_vals[sorted_vals > 0]
        if len(sorted_vals) < 10:
            continue
        ranks = np.arange(1, len(sorted_vals) + 1)
        step = max(1, len(ranks) // 5_000)
        ax.loglog(ranks[::step], sorted_vals[::step], ".", markersize=1.5,
                  alpha=0.6, color=color, label=layer_name)

    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    title = f"Multi-layer comparison: {metric_name}"
    if trim_pct > 0:
        title += f" ({trim_label}: middle {100 - 2*trim_pct:.0f}%)"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    _save_fig(fig, output_path, config)


# -----------------------------------------------------------------------
# 4. Channel-wise rank-frequency (overlay sampled channels, with trim)
# -----------------------------------------------------------------------
def plot_channel_rank_frequency(
    channel_data: np.ndarray,
    layer_name: str,
    output_path: Path,
    config: OutputConfig,
    trim_pct: float = 0,
    trim_label: str = "full",
    max_channels_to_plot: int = 16,
) -> None:
    """Overlay rank-frequency curves for a subset of sampled channels."""
    n_samples, n_channels = channel_data.shape
    indices = np.linspace(0, n_channels - 1, min(max_channels_to_plot, n_channels), dtype=int)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab20(np.linspace(0, 1, len(indices)))

    for idx, color in zip(indices, colors):
        col = channel_data[:, idx]
        trimmed = _trim_percentile(col, trim_pct)
        vals = np.sort(np.abs(trimmed))[::-1]
        vals = vals[vals > 0]
        if len(vals) < 10:
            continue
        ranks = np.arange(1, len(vals) + 1)
        step = max(1, len(ranks) // 3_000)
        ax.loglog(ranks[::step], vals[::step], ".", markersize=1, alpha=0.5,
                  color=color, label=f"ch {idx}")

    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    title = f"Channel-wise rank-frequency: {layer_name}"
    if trim_pct > 0:
        title += f" ({trim_label}: middle {100 - 2*trim_pct:.0f}%)"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=8, ncol=2, markerscale=5)
    ax.grid(True, alpha=0.3)

    _save_fig(fig, output_path, config)


# -----------------------------------------------------------------------
# 5. Channel-wise histogram (overlay sampled channels, with trim)
# -----------------------------------------------------------------------
def plot_channel_histogram(
    channel_data: np.ndarray,
    layer_name: str,
    output_path: Path,
    config: OutputConfig,
    trim_pct: float = 0,
    trim_label: str = "full",
    max_channels_to_plot: int = 16,
    n_bins: int = 100,
) -> None:
    """Overlay histograms for a subset of sampled channels."""
    n_samples, n_channels = channel_data.shape
    indices = np.linspace(0, n_channels - 1, min(max_channels_to_plot, n_channels), dtype=int)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab20(np.linspace(0, 1, len(indices)))

    for idx, color in zip(indices, colors):
        col = channel_data[:, idx]
        trimmed = _trim_percentile(col, trim_pct)
        vals = trimmed[np.isfinite(trimmed)]
        if len(vals) < 10:
            continue
        ax.hist(vals, bins=n_bins, density=True, alpha=0.3,
                color=color, label=f"ch {idx}")

    ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    title = f"Channel-wise histogram: {layer_name}"
    if trim_pct > 0:
        title += f" ({trim_label}: middle {100 - 2*trim_pct:.0f}%)"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    _save_fig(fig, output_path, config)
