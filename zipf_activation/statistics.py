from __future__ import annotations

import torch
import numpy as np
from typing import Optional

from .config import StatisticsConfig, CollectionConfig

# Shared RNG seed for reservoir sampling reproducibility
_RESERVOIR_SEED = 12345


# ---------------------------------------------------------------------------
# Individual metric functions
# Each takes flat: [N, d] float32 tensor on GPU, returns [N] tensor (or [N, k]
# for channel_wise).
# ---------------------------------------------------------------------------

def metric_rms(flat: torch.Tensor, **kw) -> torch.Tensor:
    return torch.sqrt(torch.mean(flat ** 2, dim=1))


def metric_l2_norm(flat: torch.Tensor, **kw) -> torch.Tensor:
    return torch.linalg.norm(flat, ord=2, dim=1)


def metric_l1_norm(flat: torch.Tensor, **kw) -> torch.Tensor:
    return torch.linalg.norm(flat, ord=1, dim=1)


def metric_linf_norm(flat: torch.Tensor, **kw) -> torch.Tensor:
    return torch.max(torch.abs(flat), dim=1).values


def metric_variance(flat: torch.Tensor, **kw) -> torch.Tensor:
    return torch.var(flat, dim=1)


def metric_kurtosis(flat: torch.Tensor, **kw) -> torch.Tensor:
    mu = flat.mean(dim=1, keepdim=True)
    centered = flat - mu
    m2 = (centered ** 2).mean(dim=1)
    m4 = (centered ** 4).mean(dim=1)
    return m4 / (m2 ** 2 + 1e-12) - 3.0


def metric_skewness(flat: torch.Tensor, **kw) -> torch.Tensor:
    mu = flat.mean(dim=1, keepdim=True)
    centered = flat - mu
    m2 = (centered ** 2).mean(dim=1)
    m3 = (centered ** 3).mean(dim=1)
    return m3 / (m2 ** 1.5 + 1e-12)


def metric_entropy(flat: torch.Tensor, **kw) -> torch.Tensor:
    probs = torch.softmax(flat, dim=1)
    log_probs = torch.log(probs + 1e-12)
    return -(probs * log_probs).sum(dim=1)


def metric_gini(flat: torch.Tensor, **kw) -> torch.Tensor:
    abs_flat = torch.abs(flat)
    sorted_vals, _ = torch.sort(abs_flat, dim=1)
    n = flat.shape[1]
    indices = torch.arange(1, n + 1, device=flat.device, dtype=flat.dtype)
    weighted_sum = (indices * sorted_vals).sum(dim=1)
    total = abs_flat.sum(dim=1) + 1e-12
    return (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n


def metric_participation_ratio(flat: torch.Tensor, **kw) -> torch.Tensor:
    abs_flat = torch.abs(flat)
    l1 = abs_flat.sum(dim=1)
    l2_sq = (flat ** 2).sum(dim=1)
    return (l1 ** 2) / (l2_sq + 1e-12)


def metric_mean_abs(flat: torch.Tensor, **kw) -> torch.Tensor:
    return torch.mean(torch.abs(flat), dim=1)


def metric_max_value(flat: torch.Tensor, **kw) -> torch.Tensor:
    return torch.max(flat, dim=1).values


def metric_hoyer(flat: torch.Tensor, **kw) -> torch.Tensor:
    d = flat.shape[1]
    sqrt_d = d ** 0.5
    abs_flat = torch.abs(flat)
    l1 = abs_flat.sum(dim=1)
    l2 = torch.linalg.norm(flat, ord=2, dim=1)
    return (sqrt_d - l1 / (l2 + 1e-12)) / (sqrt_d - 1.0)


def metric_channel_wise(flat: torch.Tensor, channel_indices: Optional[list[int]] = None, **kw) -> torch.Tensor:
    if channel_indices is None:
        return flat
    idx = torch.tensor(channel_indices, device=flat.device, dtype=torch.long)
    return flat[:, idx]


METRIC_FUNCTIONS = {
    "rms": metric_rms,
    "l2_norm": metric_l2_norm,
    "l1_norm": metric_l1_norm,
    "linf_norm": metric_linf_norm,
    "variance": metric_variance,
    "kurtosis": metric_kurtosis,
    "skewness": metric_skewness,
    "entropy": metric_entropy,
    "gini": metric_gini,
    "participation_ratio": metric_participation_ratio,
    "mean_abs": metric_mean_abs,
    "max_value": metric_max_value,
    "hoyer": metric_hoyer,
    "channel_wise": metric_channel_wise,
}


class StatisticsAccumulator:
    """Accumulates per-token scalar statistics across batches for one collection
    point.  All heavy computation happens on GPU; only reduced scalars are moved
    to CPU and appended to lists.
    """

    def __init__(
        self,
        metrics: list[str],
        hidden_dim: int,
        collection_cfg: CollectionConfig,
    ):
        self.metrics = metrics
        self.hidden_dim = hidden_dim
        self.data: dict[str, list[torch.Tensor]] = {m: [] for m in metrics}

        # Determine which channels to sample for channel_wise metric
        rng = np.random.RandomState(collection_cfg.channel_seed)
        k = min(collection_cfg.channel_sample_size, hidden_dim)
        self.channel_indices: list[int] = sorted(
            rng.choice(hidden_dim, size=k, replace=False).tolist()
        )

    def add_batch(self, flat: torch.Tensor) -> None:
        """flat: [N, d] on GPU.  Computes all metrics, appends CPU results."""
        flat = flat.float()
        for metric_name in self.metrics:
            fn = METRIC_FUNCTIONS[metric_name]
            values = fn(flat, channel_indices=self.channel_indices)
            self.data[metric_name].append(values.cpu())

    def finalize(self) -> dict[str, np.ndarray]:
        """Concatenate across batches and return numpy arrays."""
        out = {}
        for name, tensors in self.data.items():
            if tensors:
                out[name] = torch.cat(tensors, dim=0).numpy()
            else:
                out[name] = np.array([])
        return out


class StreamingStatisticsAccumulator:
    """Accumulates per-token statistics using reservoir sampling (Algorithm R).

    Pre-allocates fixed-size numpy arrays and never stores more than
    `reservoir_size` samples per metric, keeping memory bounded regardless
    of how many tokens are processed.
    """

    def __init__(
        self,
        metrics: list[str],
        hidden_dim: int,
        collection_cfg: CollectionConfig,
        reservoir_size: int = 1_000_000,
    ):
        self.metrics = metrics
        self.hidden_dim = hidden_dim
        self.reservoir_size = reservoir_size
        self.rng = np.random.RandomState(_RESERVOIR_SEED)

        # Determine channel_wise shape
        k = min(collection_cfg.channel_sample_size, hidden_dim)
        rng_ch = np.random.RandomState(collection_cfg.channel_seed)
        self.channel_indices: list[int] = sorted(
            rng_ch.choice(hidden_dim, size=k, replace=False).tolist()
        )

        # Pre-allocate reservoirs
        self.reservoirs: dict[str, np.ndarray] = {}
        self.total_seen: dict[str, int] = {}
        for m in metrics:
            if m == "channel_wise":
                self.reservoirs[m] = np.empty((reservoir_size, k), dtype=np.float32)
            else:
                self.reservoirs[m] = np.empty(reservoir_size, dtype=np.float32)
            self.total_seen[m] = 0

    def _update_reservoir(self, name: str, new_values: np.ndarray) -> None:
        """Vectorized reservoir sampling update (Algorithm R)."""
        reservoir = self.reservoirs[name]
        rs = self.reservoir_size
        total = self.total_seen[name]
        n = len(new_values)

        if total + n <= rs:
            # Still filling — copy directly
            reservoir[total:total + n] = new_values
        elif total < rs:
            # Partially filling, partially sampling
            fill = rs - total
            reservoir[total:rs] = new_values[:fill]
            # Remaining values need reservoir sampling
            remaining = new_values[fill:]
            m = len(remaining)
            # For value at global position (rs + j), accept with prob rs/(rs+j+1)
            positions = np.arange(rs, rs + m, dtype=np.int64)
            rand_vals = self.rng.randint(0, positions + 1)
            mask = rand_vals < rs
            if mask.any():
                reservoir[rand_vals[mask]] = remaining[mask]
        else:
            # Standard reservoir sampling: all values need probabilistic insert
            positions = np.arange(total, total + n, dtype=np.int64)
            rand_vals = self.rng.randint(0, positions + 1)
            mask = rand_vals < rs
            if mask.any():
                reservoir[rand_vals[mask]] = new_values[mask]

        self.total_seen[name] += n

    def add_batch(self, flat: torch.Tensor) -> None:
        """flat: [N, d] on GPU. Computes all metrics, does reservoir update."""
        flat = flat.float()
        for metric_name in self.metrics:
            fn = METRIC_FUNCTIONS[metric_name]
            values = fn(flat, channel_indices=self.channel_indices)
            values_np = values.cpu().numpy()
            self._update_reservoir(metric_name, values_np)

    def finalize(self) -> dict[str, np.ndarray]:
        """Return the filled portion of each reservoir."""
        out = {}
        for name in self.metrics:
            filled = min(self.total_seen[name], self.reservoir_size)
            if filled > 0:
                out[name] = self.reservoirs[name][:filled].copy()
            else:
                out[name] = np.array([])
        return out
