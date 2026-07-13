from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset


def collate_sft_batch(
    examples: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Stack multiple SFT examples into one batch.

    Input:
        examples:
            each example has:
            - input_ids.shape == (L,)
            - labels.shape == (L,)

    Output:
        batch:
            batch["input_ids"].shape == (B, L)
            batch["labels"].shape == (B, L)
    """

    if not isinstance(examples, list) or not examples:
        raise ValueError("examples must be a non-empty list.")
    
    input_ids_list = []
    labels_list = []

    for example in examples:
        if not isinstance(example, dict):
            raise ValueError("each example must be a dict.")
        
        if "input_ids" not in example or "labels" not in example:
            raise ValueError("each example must contain input_ids and labels.")
        
        input_ids = example["input_ids"]
        labels = example["labels"]

        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("example['input_ids'] must be a torch.Tensor.")

        if not isinstance(labels, torch.Tensor):
            raise TypeError("example['labels'] must be a torch.Tensor.")

        if input_ids.ndim != 1:
            raise ValueError("example['input_ids'] must have shape (L,).")

        if labels.ndim != 1:
            raise ValueError("example['labels'] must have shape (L,).")

        if input_ids.shape != labels.shape:
            raise ValueError("input_ids and labels must have the same shape.")
        
        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return {
        "input_ids": torch.stack(input_ids_list, dim=0),
        "labels": torch.stack(labels_list, dim=0),
    }


def build_sft_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """
    Build the minimal SFT dataloader.

    Input:
        dataset: torch.utils.data.Dataset
        batch_size: int
        shuffle: bool

    Output:
        dataloader: torch.utils.data.DataLoader
    """

    if not isinstance(dataset, Dataset):
        raise TypeError("dataset must be a torch.utils.data.Dataset.")

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    if not isinstance(shuffle, bool):
        raise TypeError("shuffle must be a bool.")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_sft_batch,
    )