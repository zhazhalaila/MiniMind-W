from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn

from scratch_dpo.loss import dpo_loss, logits_to_log_probs


def move_dpo_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: Union[str, torch.device],
) -> Dict[str, torch.Tensor]:
    """
    Move all DPO batch tensors to target device.

    Input:
        batch: dict[str, torch.Tensor]
            each value shape: (B, L)
        device: str | torch.device

    Output:
        moved_batch: dict[str, torch.Tensor]
            same shapes as input
    """

    return {
        key: value.to(device)
        for key, value in batch.items()
    }


def concat_chosen_rejected_batch(
    batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Concatenate chosen and rejected tensors along batch dimension.

    Input:
        batch["x_chosen"].shape == (B, L)
        batch["y_chosen"].shape == (B, L)
        batch["mask_chosen"].shape == (B, L)
        batch["x_rejected"].shape == (B, L)
        batch["y_rejected"].shape == (B, L)
        batch["mask_rejected"].shape == (B, L)

    Output:
        x: torch.Tensor
            Shape: (2B, L)
        y: torch.Tensor
            Shape: (2B, L)
        mask: torch.Tensor
            Shape: (2B, L)
    """

    x = torch.cat(
        [batch["x_chosen"], batch["x_rejected"]],
        dim=0,
    )

    y = torch.cat(
        [batch["y_chosen"], batch["y_rejected"]],
        dim=0,
    )

    mask = torch.cat(
        [batch["mask_chosen"], batch["mask_rejected"]],
        dim=0,
    )

    return x, y, mask

def compute_dpo_train_loss(
    policy_model: nn.Module,
    ref_model: nn.Module,
    batch: Dict[str, torch.Tensor],
    beta: float,
) -> Dict[str, torch.Tensor]:
    """
    Run policy/ref forward passes and compute DPO training losses.

    Input:
        policy_model: trainable model
        ref_model: frozen reference model
        batch: dict[str, torch.Tensor]
            each value shape: (B, L)
        beta: float

    Output:
        losses: dict[str, torch.Tensor]
            losses["loss"].shape == ()
            losses["dpo_loss"].shape == ()
            losses["aux_loss"].shape == ()
    """

    x, y, mask = concat_chosen_rejected_batch(batch)

    policy_outputs = policy_model(input_ids=x)
    policy_log_probs = logits_to_log_probs(
        logits=policy_outputs.logits,
        labels=y,
    )

    # notice, we not modify the parameters of ref model
    with torch.no_grad():
        ref_outputs = ref_model(input_ids=x)
        ref_log_probs = logits_to_log_probs(
            logits=ref_outputs.logits,
            labels=y,
        )

    dpo = dpo_loss(
        ref_log_probs=ref_log_probs,
        policy_log_probs=policy_log_probs,
        mask=mask,
        beta=beta,
    )

    # MoE router balance loss
    aux_loss = getattr(policy_outputs, "aux_loss", None)
    if aux_loss is None:
        aux_loss = dpo.new_zeros(())

    loss = dpo + aux_loss

    return {
        "loss": loss,
        "dpo_loss": dpo,
        "aux_loss": aux_loss,
    }


def train_dpo_one_step(
    policy_model: nn.Module,
    ref_model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer,
    device: Union[str, torch.device],
    beta: float,
) -> float:
    """
    Complete one minimal DPO optimization step.

    Output:
        loss_value: float
    """

    policy_model.train()
    ref_model.eval()

    batch = move_dpo_batch_to_device(batch, device)

    optimizer.zero_grad()

    losses = compute_dpo_train_loss(
        policy_model=policy_model,
        ref_model=ref_model,
        batch=batch,
        beta=beta,
    )

    loss = losses["loss"]
    loss.backward()
    optimizer.step()

    return float(loss.item())


def run_dpo_train_loop(
    policy_model: nn.Module,
    ref_model: nn.Module,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    optimizer,
    device: Union[str, torch.device],
    max_steps: int,
    beta: float,
    log_every: int = 1,
    save_every: Optional[int] = None,
    save_dir: Optional[str] = None,
) -> List[float]:
    """
    Run a minimal DPO training loop.

    Output:
        loss_history: list[float]
            len(loss_history) == max_steps
    """

    loss_history: list[float] = []

    step = 0
    for batch in dataloader:
        loss_value = train_dpo_one_step(
            policy_model=policy_model,
            ref_model=ref_model,
            batch=batch,
            optimizer=optimizer,
            device=device,
            beta=beta,
        )

        step += 1
        loss_history.append(loss_value)

        if log_every > 0 and step % log_every == 0:
            print(f"step={step} loss={loss_value:.6f}")

        if (
            save_every is not None
            and save_dir is not None
            and save_every > 0
            and step % save_every == 0
        ):
            save_path = Path(save_dir) / "latest_state.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": policy_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                },
                save_path,
            )

        if step >= max_steps:
            break

    return loss_history
