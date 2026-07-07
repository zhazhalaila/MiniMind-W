from __future__ import annotations

import argparse
from typing import Any, Dict

import torch
from torch import nn


def add_moe_parser_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """
    Extend one pretrain parser with MoE-specific arguments.

    Input:
        parser: argparse.ArgumentParser

    Output:
        parser: argparse.ArgumentParser

    Suggested arguments:
        --use_moe
        --num_experts
        --num_experts_per_tok
        --moe_intermediate_size
        --router_aux_loss_coef
    """

    parser.add_argument(
        "--use_moe",
        type=int,
        default=0,
        help="Enable MoE feed-forward layers. 0 = dense, 1 = MoE.",
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=4,
        help="Total number of experts in the MoE layer.",
    )
    parser.add_argument(
        "--num_experts_per_tok",
        type=int,
        default=1,
        help="Number of top-k experts selected for each token.",
    )
    parser.add_argument(
        "--moe_intermediate_size",
        type=int,
        default=None,
        help="Intermediate FFN size inside each expert. Defaults to intermediate_size.",
    )
    parser.add_argument(
        "--router_aux_loss_coef",
        type=float,
        default=5e-4,
        help="Coefficient for the router auxiliary load-balancing loss.",
    )
    return parser


def build_moe_kwargs_from_args(
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Extract MoE-related config fields from parsed args.

    Input:
        args: argparse.Namespace

    Output:
        moe_kwargs: dict[str, Any]
            Suggested keys:
            - "use_moe"
            - "num_experts"
            - "num_experts_per_tok"
            - "moe_intermediate_size"
            - "router_aux_loss_coef"
    """

    moe_intermediate_size = args.moe_intermediate_size
    if moe_intermediate_size is None:
        moe_intermediate_size = args.intermediate_size

    return {
        "use_moe": bool(args.use_moe),
        "num_experts": args.num_experts,
        "num_experts_per_tok": args.num_experts_per_tok,
        "moe_intermediate_size": moe_intermediate_size,
        "router_aux_loss_coef": args.router_aux_loss_coef,
    }


def build_moe_weight_name(
    save_weight: str,
    hidden_size: int,
    use_moe: bool,
) -> str:
    """
    Build the canonical Dense / MoE checkpoint base name.

    Input:
        save_weight: str
        hidden_size: int
        use_moe: bool

    Output:
        weight_name: str

    Example:
        Dense -> "pretrain_768"
        MoE   -> "pretrain_768_moe"
    """

    suffix = "_moe" if use_moe else ""
    return f"{save_weight}_{hidden_size}{suffix}"


def collect_router_aux_loss(
    model: nn.Module,
) -> torch.Tensor:
    """
    Collect router auxiliary loss from one MoE model.

    Input:
        model: torch.nn.Module

    Output:
        router_aux_loss: torch.Tensor
            Shape: ()

    Suggested behavior:
        - iterate through decoder blocks
        - find MoE submodules
        - sum their aux losses
        - return scalar zero if the model is Dense
    """

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    
    aux_loss = torch.zeros((), device=device)

    for module in model.modules():
        if hasattr(module, "aux_loss"):
            value = getattr(module, "aux_loss")
            if isinstance(value, torch.Tensor):
                aux_loss = aux_loss + value

    return aux_loss


def combine_lm_and_router_loss(
    lm_loss: torch.Tensor,
    router_aux_loss: torch.Tensor,
) -> torch.Tensor:
    """
    Combine the main LM loss and the router auxiliary loss.

    Input:
        lm_loss: torch.Tensor
            Shape: ()
        router_aux_loss: torch.Tensor
            Shape: ()

    Output:
        total_loss: torch.Tensor
            Shape: ()
    """

    return lm_loss + router_aux_loss


def build_moe_smoke_test_kwargs() -> Dict[str, Any]:
    """
    Build one tiny MoE config dict for smoke tests.

    Output:
        moe_kwargs: dict[str, Any]

    Suggested values:
        - use_moe = True
        - num_experts = 2
        - num_experts_per_tok = 1
        - moe_intermediate_size = 16
        - router_aux_loss_coef = 5e-4
    """
    return {
        "use_moe": True,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 16,
        "router_aux_loss_coef": 5e-4,
    }