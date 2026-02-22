from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .config import OutputConfig
from .fitting import PowerLawFitResult


def _save_fig(fig: plt.Figure, path: Path, config: OutputConfig) -> None:
    formats = [config.plot_format] if config.plot_format != "both" else ["pdf", "png"]
    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"), format=fmt, dpi=config.plot_dpi,
                    bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------
# 1. Rank-frequency log-log plot
# -----------------------------------------------------------------------
def plot_rank_frequency(
    values: np.ndarray,
    fit_result: PowerLawFitResult,
    output_path: Path,
    config: OutputConfig,
) -> None:
    sorted_vals = np.sort(np.abs(values))[::-1]
    sorted_vals = sorted_vals[sorted_vals > 0]
    if len(sorted_vals) < 10:
        return

    ranks = np.arange(1, len(sorted_vals) + 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot every k-th point for speed when N is large
    step = max(1, len(ranks) // 10_000)
    ax.loglog(ranks[::step], sorted_vals[::step], ".", markersize=1, alpha=0.4, label="Data")

    # Overlay power law fit line from log-log regression
    if fit_result.log_log_r_squared > 0:
        fit_line = np.exp(fit_result.log_log_intercept) * ranks.astype(float) ** fit_result.log_log_slope
        ax.loglog(ranks[::step], fit_line[::step], "r-", linewidth=1.5,
                  label=f"Fit (slope={fit_result.log_log_slope:.2f}, R²={fit_result.log_log_r_squared:.3f})")

    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    title = f"{fit_result.layer_name} / {fit_result.metric_name}"
    ax.set_title(title, fontsize=14)
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
# 3. Exponent dashboard (heatmap)
# -----------------------------------------------------------------------
def plot_exponent_dashboard(
    fit_results: dict[str, dict[str, PowerLawFitResult]],
    output_path: Path,
    config: OutputConfig,
) -> None:
    """Heatmap of alpha exponents across (layer, metric)."""
    layer_names = sorted(fit_results.keys())
    metric_names: list[str] = []
    for lr in fit_results.values():
        for mn in lr:
            if mn not in metric_names:
                metric_names.append(mn)

    data = np.full((len(metric_names), len(layer_names)), np.nan)
    for j, layer in enumerate(layer_names):
        for i, metric in enumerate(metric_names):
            if metric in fit_results[layer]:
                data[i, j] = fit_results[layer][metric].alpha

    fig, ax = plt.subplots(figsize=(max(8, len(layer_names) * 2), max(6, len(metric_names) * 0.5)))
    sns.heatmap(data, annot=True, fmt=".2f", xticklabels=layer_names,
                yticklabels=metric_names, cmap="YlOrRd", ax=ax)
    ax.set_title("Power Law Exponent (alpha) by Layer and Metric", fontsize=14)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Metric")

    _save_fig(fig, output_path, config)


def plot_slope_dashboard(
    fit_results: dict[str, dict[str, PowerLawFitResult]],
    output_path: Path,
    config: OutputConfig,
) -> None:
    """Heatmap of log-log slopes across (layer, metric)."""
    layer_names = sorted(fit_results.keys())
    metric_names: list[str] = []
    for lr in fit_results.values():
        for mn in lr:
            if mn not in metric_names:
                metric_names.append(mn)

    data = np.full((len(metric_names), len(layer_names)), np.nan)
    for j, layer in enumerate(layer_names):
        for i, metric in enumerate(metric_names):
            if metric in fit_results[layer]:
                data[i, j] = fit_results[layer][metric].log_log_slope

    fig, ax = plt.subplots(figsize=(max(8, len(layer_names) * 2), max(6, len(metric_names) * 0.5)))
    sns.heatmap(data, annot=True, fmt=".2f", xticklabels=layer_names,
                yticklabels=metric_names, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Log-Log Slope by Layer and Metric", fontsize=14)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Metric")

    _save_fig(fig, output_path, config)


# -----------------------------------------------------------------------
# 4. Distribution comparison (CCDF)
# -----------------------------------------------------------------------
def plot_distribution_comparison(
    values: np.ndarray,
    fit_result: PowerLawFitResult,
    output_path: Path,
    config: OutputConfig,
) -> None:
    """Empirical CCDF with annotation of fit quality."""
    abs_values = np.sort(np.abs(values))[::-1]
    abs_values = abs_values[abs_values > 0]
    if len(abs_values) < 10:
        return

    ccdf = np.arange(1, len(abs_values) + 1) / len(abs_values)

    fig, ax = plt.subplots(figsize=(8, 6))
    step = max(1, len(abs_values) // 10_000)
    ax.loglog(abs_values[::step], ccdf[::step], ".", markersize=1, alpha=0.4, label="Empirical CCDF")

    # Annotations
    text_lines = [
        f"alpha = {fit_result.alpha:.2f} (Clauset)",
        f"xmin = {fit_result.xmin:.4g}",
        f"KS = {fit_result.ks_statistic:.4f}",
        f"n_tail = {fit_result.n_tail}",
        f"slope = {fit_result.log_log_slope:.2f} (OLS)",
        f"R² = {fit_result.log_log_r_squared:.3f}",
    ]
    for dist, comp in fit_result.comparisons.items():
        r, p = comp["R"], comp["p"]
        text_lines.append(f"vs {dist}: R={r:.2f}, p={p:.3f}")

    ax.text(0.02, 0.02, "\n".join(text_lines), transform=ax.transAxes,
            fontsize=8, verticalalignment="bottom", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("P(X >= x)", fontsize=12)
    title = f"CCDF: {fit_result.layer_name} / {fit_result.metric_name}"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    _save_fig(fig, output_path, config)


# -----------------------------------------------------------------------
# 5. Channel-wise heatmap
# -----------------------------------------------------------------------
def plot_channel_heatmap(
    channel_fit_results: dict[str, list[PowerLawFitResult]],
    output_path: Path,
    config: OutputConfig,
) -> None:
    """Heatmap of alpha exponents across sampled channels and layers."""
    layer_names = sorted(channel_fit_results.keys())
    if not layer_names:
        return

    n_channels = len(channel_fit_results[layer_names[0]])
    data = np.full((n_channels, len(layer_names)), np.nan)

    for j, layer in enumerate(layer_names):
        for i, fr in enumerate(channel_fit_results[layer]):
            data[i, j] = fr.log_log_slope

    fig, ax = plt.subplots(figsize=(max(8, len(layer_names) * 2), max(6, n_channels * 0.15)))
    sns.heatmap(data, xticklabels=layer_names, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Channel-wise Log-Log Slope by Layer", fontsize=14)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Sampled Channel Index")

    _save_fig(fig, output_path, config)
