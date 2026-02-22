from __future__ import annotations

import argparse
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3-4B-Base"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True


@dataclass
class DataConfig:
    name: str = "Skylion007/openwebtext"
    subset: Optional[str] = None  # e.g. "wikitext-2-raw-v1" for wikitext
    split: str = "train"
    streaming: bool = True
    max_tokens: int = 10_000_000
    batch_size: int = 32
    max_seq_len: int = 2048
    num_workers: int = 0


@dataclass
class CollectionConfig:
    collection_points: list[str] = field(
        default_factory=lambda: ["embed", "middle", "pre_last"]
    )
    explicit_layers: Optional[list[int]] = None
    channel_sample_size: int = 64
    channel_seed: int = 42


@dataclass
class StatisticsConfig:
    metrics: list[str] = field(
        default_factory=lambda: [
            "channel_wise",
            "rms",
            "l2_norm",
            "l1_norm",
            "linf_norm",
            "variance",
            "kurtosis",
            "skewness",
            "entropy",
            "gini",
            "participation_ratio",
            "mean_abs",
            "max_value",
            "hoyer",
        ]
    )


@dataclass
class FittingConfig:
    compare_distributions: list[str] = field(
        default_factory=lambda: ["lognormal", "exponential", "truncated_power_law"]
    )
    max_samples_for_fit: int = 50_000
    fit_seed: int = 42


@dataclass
class OutputConfig:
    results_dir: str = "results"
    plot_format: str = "pdf"
    plot_dpi: int = 300
    save_statistics: bool = True


@dataclass
class ZipfActivationConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    fitting: FittingConfig = field(default_factory=FittingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _apply_dict(dc: object, d: dict) -> None:
    for k, v in d.items():
        if hasattr(dc, k):
            attr = getattr(dc, k)
            if isinstance(attr, (ModelConfig, DataConfig, CollectionConfig,
                                 StatisticsConfig, FittingConfig, OutputConfig)):
                _apply_dict(attr, v)
            else:
                setattr(dc, k, v)


def load_config(yaml_path: Optional[str] = None, overrides: Optional[dict] = None) -> ZipfActivationConfig:
    cfg = ZipfActivationConfig()
    if yaml_path is not None:
        with open(yaml_path) as f:
            d = yaml.safe_load(f) or {}
        _apply_dict(cfg, d)
    if overrides:
        _apply_dict(cfg, overrides)
    return cfg


def config_from_cli() -> ZipfActivationConfig:
    parser = argparse.ArgumentParser(description="ZipfActivation")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--model.name", dest="model_name", type=str, default=None)
    parser.add_argument("--data.name", dest="data_name", type=str, default=None)
    parser.add_argument("--data.subset", dest="data_subset", type=str, default=None)
    parser.add_argument("--data.max_tokens", dest="data_max_tokens", type=int, default=None)
    parser.add_argument("--data.batch_size", dest="data_batch_size", type=int, default=None)
    parser.add_argument("--data.max_seq_len", dest="data_max_seq_len", type=int, default=None)
    parser.add_argument("--output.results_dir", dest="output_results_dir", type=str, default=None)
    parser.add_argument("--output.plot_format", dest="output_plot_format", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply CLI overrides
    overrides: dict = {}
    if args.model_name:
        overrides.setdefault("model", {})["name"] = args.model_name
    if args.data_name:
        overrides.setdefault("data", {})["name"] = args.data_name
    if args.data_subset:
        overrides.setdefault("data", {})["subset"] = args.data_subset
    if args.data_max_tokens is not None:
        overrides.setdefault("data", {})["max_tokens"] = args.data_max_tokens
    if args.data_batch_size is not None:
        overrides.setdefault("data", {})["batch_size"] = args.data_batch_size
    if args.data_max_seq_len is not None:
        overrides.setdefault("data", {})["max_seq_len"] = args.data_max_seq_len
    if args.output_results_dir:
        overrides.setdefault("output", {})["results_dir"] = args.output_results_dir
    if args.output_plot_format:
        overrides.setdefault("output", {})["plot_format"] = args.output_plot_format

    if overrides:
        _apply_dict(cfg, overrides)

    return cfg
