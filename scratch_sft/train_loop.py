from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn


def compute_sft_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Run one forward pass and return SFT scalar loss.

    Input:
        model: nn.Module
        batch:
            batch["input_ids"].shape == (B, L)
            batch["labels"].shape == (B, L)

    Output:
        loss: torch.Tensor
            Shape: ()
    """

    if not isinstance(batch, dict):
        raise TypeError("batch must be a dict.")

    if "input_ids" not in batch or "labels" not in batch:
        raise ValueError("batch must contain input_ids and labels.")

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("batch['input_ids'] must be a torch.Tensor.")

    if not isinstance(labels, torch.Tensor):
        raise TypeError("batch['labels'] must be a torch.Tensor.")

    if input_ids.ndim != 2:
        raise ValueError("batch['input_ids'] must have shape (B, L).")

    if labels.ndim != 2:
        raise ValueError("batch['labels'] must have shape (B, L).")

    if input_ids.shape != labels.shape:
        raise ValueError("input_ids and labels must have the same shape.")
    
    outputs = model(
        input_ids=input_ids,
        labels=labels,
    )

    loss = outputs.loss

    if loss is None:
        raise ValueError("model output loss is None.")
    
    return loss

def train_sft_one_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer,
    device,
) -> float:
    """
    Complete one full SFT optimization step.

    Output:
        loss_value: float
    """

    model.train()

    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    optimizer.zero_grad()

    loss = compute_sft_loss(
        model=model,
        batch=batch,
    )

    loss.backward()
    optimizer.step()

    return float(loss.item())

def run_sft_train_loop(
    model: nn.Module,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    optimizer,
    device,
    max_steps: int,
    log_every: int = 1,
    save_every: Optional[int] = None,
    save_dir: Optional[str] = None,
) -> List[float]:
    """
    Run the minimal SFT training loop.

    Input:
        dataloader yields:
            batch["input_ids"].shape == (B, L)
            batch["labels"].shape == (B, L)

    Output:
        loss_history: list[float]
            len(loss_history) == max_steps
    """

    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer.")

    if not isinstance(log_every, int) or log_every <= 0:
        raise ValueError("log_every must be a positive integer.")

    if save_every is not None and (not isinstance(save_every, int) or save_every <= 0):
        raise ValueError("save_every must be a positive integer or None.")

    if save_every is not None and save_dir is None:
        raise ValueError("save_dir must be provided when save_every is set.")

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    loss_history: List[float] = []
    step = 0

    while step < max_steps:
        for batch in dataloader:
            loss_value = train_sft_one_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                device=device,
            )

            step += 1
            loss_history.append(loss_value)

            if step % log_every == 0:
                print(f"step={step} loss={loss_value:.6f}")

            if save_every is not None and step % save_every == 0:
                save_path = Path(save_dir) / f"sft_step_{step}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                    },
                    save_path,
                )

            if step >= max_steps:
                break

    return loss_history