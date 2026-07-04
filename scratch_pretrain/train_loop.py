from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Union

from pathlib import Path
import torch
from torch import nn
from scratch_pretrain.checkpoint import build_checkpoint_state, save_checkpoint

def move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: Union[str, torch.device],
) -> Dict[str, torch.Tensor]:
    """
    Move one batch dictionary onto the target device.

    Input:
        batch: dict[str, torch.Tensor]
            - batch["input_ids"]: shape (B, L)
            - batch["labels"]: shape (B, L)
        device: str | torch.device

    Output:
        moved_batch: dict[str, torch.Tensor]
            - moved_batch["input_ids"]: shape (B, L)
            - moved_batch["labels"]: shape (B, L)
    """

    moved_batch = {}

    for key, value in batch.items():
        moved_batch[key] = value.to(device)

    return moved_batch


def compute_pretrain_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Run one forward pass and return the scalar training loss.

    Input:
        model: nn.Module
        batch: dict[str, torch.Tensor]
            - batch["input_ids"]: shape (B, L)
            - batch["labels"]: shape (B, L)

    Output:
        loss: torch.Tensor
            Shape: ()

    Expected model call:
        model(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
        )
    """

    output = model(
        input_ids=batch["input_ids"],
        labels=batch["labels"],
    )

    return output.loss

def train_one_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: Union[str, torch.device],
) -> float:
    """
    Run one complete train step.

    Input:
        model: nn.Module
        batch: dict[str, torch.Tensor]
            - batch["input_ids"]: shape (B, L)
            - batch["labels"]: shape (B, L)
        optimizer: torch.optim.Optimizer
        device: str | torch.device

    Output:
        loss_value: float

    Expected internal steps:
        1. model.train()
        2. move batch to device
        3. zero gradients
        4. forward
        5. backward
        6. optimizer.step()
        7. return python float loss
    """

    model.train()
    batch = move_batch_to_device(batch, device)
    optimizer.zero_grad()
    loss = compute_pretrain_loss(model, batch)
    loss.backward()
    optimizer.step()
    return loss.item()


def run_pretrain_train_loop(
    model: nn.Module,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: Union[str, torch.device],
    max_steps: int,
    log_every: int = 1,
    save_every: Optional[int] = None,
    save_dir: Optional[str] = None,
) -> List[float]:
    """
    Run the minimal pretrain train loop.

    Input:
        model: nn.Module
        dataloader: iterable[dict[str, torch.Tensor]]
            Each batch is expected to contain:
            - batch["input_ids"]: shape (B, L)
            - batch["labels"]: shape (B, L)
        optimizer: torch.optim.Optimizer
        device: str | torch.device
        max_steps: int
        log_every: int
        save_every: int | None
        save_dir: str | None

    Output:
        loss_history: list[float]
            Expected length: max_steps

    Expected internal behavior:
        - call train_one_step for each batch
        - append each step loss into loss_history
        - optionally print logs every log_every steps
        - optionally save checkpoints every save_every steps
    """

    loss_history = []
    step = 0

    if save_every is not None and save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    while step < max_steps:
        for batch in dataloader:
            loss_value = train_one_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                device=device,
            )

            step += 1
            loss_history.append(loss_value)

            if log_every > 0 and step % log_every == 0:
                print(f"step={step}, loss={loss_value:.6f}")

            if (
                save_every is not None
                and save_dir is not None
                and save_every > 0
                and step % save_every == 0
            ):
                checkpoint = build_checkpoint_state(model, optimizer, step)
                save_path = Path(save_dir) / f"step_{step}.pt"
                save_checkpoint(checkpoint, save_path)

            if step >= max_steps:
                break

        else:
            break

    return loss_history