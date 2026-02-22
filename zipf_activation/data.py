from __future__ import annotations

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from typing import Iterator

from .config import DataConfig


class PackedTokenDataset(IterableDataset):
    """Streams a HuggingFace text dataset, tokenizes on the fly, and yields
    packed fixed-length sequences (no padding waste).

    Documents are concatenated into a running token buffer. When the buffer is
    large enough, a batch of [batch_size, max_seq_len] is yielded. Stops after
    max_tokens total tokens have been produced.
    """

    def __init__(self, config: DataConfig, tokenizer: PreTrainedTokenizerBase):
        self.config = config
        self.tokenizer = tokenizer

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        cfg = self.config
        ds = load_dataset(cfg.name, cfg.subset, split=cfg.split, streaming=cfg.streaming)

        buffer: list[int] = []
        total_tokens = 0
        chunk_size = cfg.batch_size * cfg.max_seq_len

        for example in ds:
            text = example.get("text", "")
            if not text:
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)

            while len(buffer) >= chunk_size:
                chunk = buffer[:chunk_size]
                buffer = buffer[chunk_size:]

                input_ids = torch.tensor(chunk, dtype=torch.long).reshape(
                    cfg.batch_size, cfg.max_seq_len
                )
                attention_mask = torch.ones_like(input_ids)

                total_tokens += chunk_size
                yield {"input_ids": input_ids, "attention_mask": attention_mask}

                if total_tokens >= cfg.max_tokens:
                    return


def create_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizerBase) -> DataLoader:
    dataset = PackedTokenDataset(config, tokenizer)
    return DataLoader(
        dataset,
        batch_size=None,  # dataset already yields full batches
        num_workers=config.num_workers,
        pin_memory=True,
    )
