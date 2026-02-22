from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .config import ModelConfig, CollectionConfig, StatisticsConfig
from .statistics import StatisticsAccumulator


def load_model(config: ModelConfig) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(
        config.name, trust_remote_code=config.trust_remote_code
    )
    dtype = getattr(torch, config.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        config.name,
        dtype=dtype,
        device_map=config.device_map,
        trust_remote_code=config.trust_remote_code,
    )
    model.eval()
    return model, tokenizer


def _get_base_model(model: PreTrainedModel) -> nn.Module:
    """Unwrap the CausalLM wrapper to get the base transformer.

    For most HuggingFace models:
      - model.model      (Llama, Qwen, Mistral, etc.)
      - model.transformer (GPT-2, GPT-Neo, etc.)
    """
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "transformer"):
        return model.transformer
    return model


def _get_embedding_module(base: nn.Module) -> nn.Module:
    for attr in ("embed_tokens", "wte", "word_embedding", "embeddings"):
        if hasattr(base, attr):
            return getattr(base, attr)
    raise ValueError(f"Cannot find embedding module in {type(base)}")


def _get_layers(base: nn.Module) -> nn.ModuleList:
    for attr in ("layers", "h", "blocks"):
        if hasattr(base, attr):
            return getattr(base, attr)
    raise ValueError(f"Cannot find layers in {type(base)}")


def _resolve_collection_points(
    model: PreTrainedModel,
    config: CollectionConfig,
) -> dict[str, nn.Module]:
    """Map symbolic collection point names to actual nn.Module instances."""
    base = _get_base_model(model)
    layers = _get_layers(base)
    num_layers = len(layers)

    name_to_module: dict[str, nn.Module] = {}

    # Handle explicit layer indices
    if config.explicit_layers:
        for idx in config.explicit_layers:
            name_to_module[f"layer_{idx}"] = layers[idx]
        return name_to_module

    # Handle symbolic names
    for point in config.collection_points:
        if point == "embed":
            name_to_module["embed"] = _get_embedding_module(base)
        elif point == "middle":
            mid = num_layers // 2
            name_to_module[f"layer_{mid}"] = layers[mid]
        elif point == "pre_last":
            name_to_module[f"layer_{num_layers - 2}"] = layers[num_layers - 2]
        elif point.startswith("layer_"):
            idx = int(point.split("_")[1])
            name_to_module[point] = layers[idx]
        else:
            raise ValueError(f"Unknown collection point: {point}")

    return name_to_module


class ActivationCollector:
    """Registers forward hooks on specified layers and collects per-token
    scalar statistics on the fly.  Never stores full activation tensors.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        collection_cfg: CollectionConfig,
        stats_cfg: StatisticsConfig,
    ):
        self.model = model
        self.hooks: list[torch.utils.hooks.RemovableHook] = []
        self.accumulators: dict[str, StatisticsAccumulator] = {}

        layer_map = _resolve_collection_points(model, collection_cfg)
        hidden_dim = model.config.hidden_size

        for name, module in layer_map.items():
            acc = StatisticsAccumulator(
                metrics=stats_cfg.metrics,
                hidden_dim=hidden_dim,
                collection_cfg=collection_cfg,
            )
            self.accumulators[name] = acc
            handle = module.register_forward_hook(self._make_hook(name))
            self.hooks.append(handle)

    def _make_hook(self, layer_name: str):
        def hook_fn(module: nn.Module, input, output):
            # Decoder layers often return a tuple (hidden_states, ...)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # hidden_states: [batch, seq_len, hidden_dim]
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            self.accumulators[layer_name].add_batch(flat)

        return hook_fn

    def remove_hooks(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
