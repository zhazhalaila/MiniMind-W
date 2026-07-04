from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn


def build_checkpoint_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
) -> Dict[str, Any]:
    """
    Build the minimal checkpoint payload.

    Input:
        model: nn.Module
        optimizer: torch.optim.Optimizer
        step: int

    Output:
        checkpoint: dict[str, Any]
            Expected keys:
            - "model"
            - "optimizer"
            - "step"
    """

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    
    return checkpoint


def save_checkpoint(checkpoint: Dict[str, Any], save_path: str) -> None:
    """
    Save one checkpoint to local disk.

    Input:
        checkpoint: dict[str, Any]
        save_path: str

    Output:
        none

    Expected side effect:
        After this function runs, a file should exist at save_path.
    """

    torch.save(checkpoint, save_path)
