from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


def collate_pretrain_batch(
    examples: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Stack single-sample tensors into one train batch.

    Input:
        examples: list[tuple[torch.Tensor, torch.Tensor]]
            Each tuple is:
            - input_ids: shape (L,)
            - labels: shape (L,)

    Output:
        batch: dict[str, torch.Tensor]
            - batch["input_ids"]: shape (B, L)
            - batch["labels"]: shape (B, L)

    Example:
        If len(examples) == 2 and each sample length is 8,
        then:
        - output["input_ids"].shape == (2, 8)
        - output["labels"].shape == (2, 8)
    """

    input_ids_list = []
    labels_list = []

    for input_ids, labels in examples:
        input_ids_list.append(input_ids)
        labels_list.append(labels)

    batch_input_ids = torch.stack(input_ids_list, dim=0)
    batch_labels = torch.stack(labels_list, dim=0)

    return {
        "input_ids": batch_input_ids,
        "labels": batch_labels,
    }


def build_pretrain_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    """
    Build the minimal pretrain dataloader.

    Input:
        dataset: torch.utils.data.Dataset
            __getitem__ is expected to return:
            - input_ids: shape (L,)
            - labels: shape (L,)
        batch_size: int
        shuffle: bool
        num_workers: int
        generator: torch.Generator | None

    Output:
        dataloader: torch.utils.data.DataLoader
            Iterating one batch should produce:
            - batch["input_ids"]: shape (B, L)
            - batch["labels"]: shape (B, L)

    Suggested implementation:
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            generator=generator,
            collate_fn=collate_pretrain_batch,
        )
    """

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        collate_fn=collate_pretrain_batch,
    )
