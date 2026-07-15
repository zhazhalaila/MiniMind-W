from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset


DPO_BATCH_KEYS = (
    "x_chosen",
    "y_chosen",
    "mask_chosen",
    "x_rejected",
    "y_rejected",
    "mask_rejected",
)


def collate_dpo_batch(
    examples: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Stack multiple DPO examples into one batch.

    Input:
        examples:
            each example contains six tensors:
            - x_chosen.shape == (L,)
            - y_chosen.shape == (L,)
            - mask_chosen.shape == (L,)
            - x_rejected.shape == (L,)
            - y_rejected.shape == (L,)
            - mask_rejected.shape == (L,)

    Output:
        batch:
            each value has shape (B, L)
    """

    batch: dict[str, torch.Tensor] = {}

    for key in DPO_BATCH_KEYS:
        batch[key] = torch.stack(
            [example[key] for example in examples],
            dim=0,
        )

    return batch


def build_dpo_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """
    Build a DPO dataloader.

    Input:
        dataset: torch.utils.data.Dataset
        batch_size: int
        shuffle: bool

    Output:
        dataloader: torch.utils.data.DataLoader
    """

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_dpo_batch,
    )