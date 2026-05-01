"""Dataset and dataloader utilities."""

from __future__ import annotations

import glob
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .config import DataConfig


class ArrowTokenDataset(Dataset):
    """Loads tokenized Hugging Face Arrow shards with an input_ids column."""

    def __init__(self, dataset_folder: str, input_column: str = "input_ids") -> None:
        from datasets import Dataset as HFDataset
        from datasets import concatenate_datasets

        arrow_files = sorted(glob.glob(os.path.join(dataset_folder, "*.arrow")))
        if not arrow_files:
            raise ValueError(f"No .arrow files found in {dataset_folder}")

        datasets = [HFDataset.from_file(path) for path in arrow_files]
        self.dataset = concatenate_datasets(datasets)
        self.input_column = input_column

        if input_column not in self.dataset.column_names:
            raise ValueError(
                f"Column {input_column!r} not found. Available columns: "
                f"{', '.join(self.dataset.column_names)}"
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor(self.dataset[idx][self.input_column])}


def build_train_dataloader(
    dataset: Dataset,
    config: DataConfig,
    batch_size: int,
    world_size: int,
    rank: int,
) -> DataLoader:
    sampler = None
    shuffle = config.shuffle

    if config.use_distributed_sampler and world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=max(0, rank),
            shuffle=config.shuffle,
            drop_last=config.drop_last,
        )
        shuffle = False

    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "drop_last": config.drop_last,
    }
    if config.num_workers > 0:
        kwargs["prefetch_factor"] = config.prefetch_factor
        kwargs["persistent_workers"] = config.persistent_workers

    return DataLoader(dataset, **kwargs)
