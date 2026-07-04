from __future__ import annotations

import torch
from torch import nn


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """
    Build the minimal optimizer for pretrain.

    Input:
        model: nn.Module
        learning_rate: float
        weight_decay: float

    Output:
        optimizer: torch.optim.Optimizer

    Suggested implementation:
        torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    """

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    return optimizer
