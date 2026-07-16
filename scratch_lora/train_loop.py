from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn

from scratch_lora.lora import iter_lora_modules


def compute_lora_sft_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Run one LoRA SFT forward pass and return scalar loss.

    Input:
        batch["input_ids"].shape == (B, L)
        batch["labels"].shape == (B, L)

    Output:
        loss: torch.Tensor
            Shape: ()
    """

    if "input_ids" not in batch:
        raise KeyError("batch must contain 'input_ids'.")

    if "labels" not in batch:
        raise KeyError("batch must contain 'labels'.")

    outputs = model(
        input_ids=batch["input_ids"],
        labels=batch["labels"],
    )

    loss = outputs.loss

    if loss is None:
        raise ValueError("model output loss is None.")

    return loss


def train_lora_one_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer,
    device,
) -> float:
    """
    Complete one minimal LoRA optimization step.

    Output:
        loss_value: float
    """

    model.train()

    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    optimizer.zero_grad()

    loss = compute_lora_sft_loss(model, batch)
    loss.backward()

    optimizer.step()

    return float(loss.item())


def run_lora_train_loop(
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
    Run the minimal LoRA training loop.

    Input:
        dataloader yields:
            batch["input_ids"].shape == (B, L)
            batch["labels"].shape == (B, L)

    Output:
        loss_history: list[float]
    """

    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer.")

    if not isinstance(log_every, int) or log_every <= 0:
        raise ValueError("log_every must be a positive integer.")

    if save_every is not None and (not isinstance(save_every, int) or save_every <= 0):
        raise ValueError("save_every must be a positive integer or None.")

    if save_every is not None and save_dir is None:
        raise ValueError("save_dir is required when save_every is set.")

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    loss_history: List[float] = []
    step = 0

    while step < max_steps:
        for batch in dataloader:
            loss_value = train_lora_one_step(
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
                save_path = Path(save_dir) / "latest.pt"
                save_lora_training_state(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    path=save_path,
                )

            if step >= max_steps:
                break

    return loss_history


def save_lora_training_state(
    model: nn.Module,
    optimizer,
    step: int,
    path: str | Path,
) -> None:
    
    """
    Save LoRA resume state.

    Output file:
        dict with:
            "model": LoRA-only or model state depending on implementation choice.
            "optimizer": optimizer.state_dict()
            "step": int
    """
    if not isinstance(model, nn.Module):
        raise ValueError("model must be an nn.Module.")

    if not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    raw_model = getattr(model, "_orig_mod", model)

    lora_state = {}

    for name, module in iter_lora_modules(raw_model):
        clean_name = name[7:] if name.startswith("module.") else name

        for key, value in module.lora.state_dict().items():
            lora_state[f"{clean_name}.lora.{key}"] = value.detach().cpu()

    checkpoint = {
        "model": lora_state,
        "optimizer": optimizer.state_dict(),
        "step": step,
    }

    torch.save(checkpoint, path)