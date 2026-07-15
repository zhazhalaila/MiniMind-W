from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

def logits_to_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Gather log probabilities of target labels from logits.

    Input:
        logits: torch.Tensor
            Shape: (B, L, V)
        labels: torch.Tensor
            Shape: (B, L)

    Output:
        log_probs_per_token: torch.Tensor
            Shape: (B, L)
    """

    log_probs = F.log_softmax(logits, dim=-1)

    # for each label token (correct answer, regardless chosen or reject), we get it's probability from the forward output
    # shape (B, L)
    label_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    return label_log_probs


def masked_sequence_log_probs(
    log_probs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Sum token log probabilities over assistant-mask positions.

    Input:
        log_probs: torch.Tensor
            Shape: (B, L)
        mask: torch.Tensor
            Shape: (B, L)

    Output:
        sequence_log_probs: torch.Tensor
            Shape: (B,)
    """

    return (log_probs * mask).sum(dim=-1)


def split_chosen_rejected(values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split concatenated chosen+rejected batch values.

    Input:
        values: torch.Tensor
            Shape: (2B,)

    Output:
        chosen_values: torch.Tensor
            Shape: (B,)
        rejected_values: torch.Tensor
            Shape: (B,)
    """

    if values.size(0) % 2 != 0:
        raise ValueError("values first dimension must be even.")
    
    half = values.size(0) // 2
    chosen_values = values[:half]
    rejected_values = values[half:]

    return chosen_values, rejected_values


def dpo_loss(
    ref_log_probs: torch.Tensor,
    policy_log_probs: torch.Tensor,
    mask: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """
    Compute MiniMind-style DPO loss on concatenated chosen+rejected batches.

    Input:
        ref_log_probs: torch.Tensor
            Shape: (2B, L)
        policy_log_probs: torch.Tensor
            Shape: (2B, L)
        mask: torch.Tensor
            Shape: (2B, L)
        beta: float

    Output:
        loss: torch.Tensor
            Shape: ()
    """

    ref_sequence_log_probs = masked_sequence_log_probs(ref_log_probs, mask)
    policy_sequence_log_probs = masked_sequence_log_probs(policy_log_probs, mask)

    chosen_ref_log_probs, rejected_ref_log_probs = split_chosen_rejected(
        ref_sequence_log_probs
    )

    chosen_policy_log_probs, rejected_policy_log_probs = split_chosen_rejected(
        policy_sequence_log_probs
    )

    pi_logratios = chosen_policy_log_probs - rejected_policy_log_probs
    ref_logratios = chosen_ref_log_probs - rejected_ref_log_probs

    # prefer chosen, but not deviate far away from origin model
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()

    return loss