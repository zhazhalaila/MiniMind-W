from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn


class LoRA(nn.Module):
    """
    Low-rank adapter.

    Math:
        delta = B(A(x))
        output shape follows the wrapped Linear output shape.

    Input to forward:
        x: torch.Tensor
            Shape: (..., in_features)

    Output from forward:
        torch.Tensor
            Shape: (..., out_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
    ) -> None:
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(x))


def parse_target_modules(target_modules: Optional[str]) -> Optional[List[str]]:
    """
    Parse comma-separated target module name fragments.

    Input:
        target_modules: str | None

    Output:
        list[str] | None
    """

    if target_modules is None:
        return None

    if not isinstance(target_modules, str):
        raise ValueError("target_modules must be a string or None.")

    parts = [
        item.strip()
        for item in target_modules.split(",")
        if item.strip()
    ]

    if not parts:
        return None

    return parts


def should_apply_lora(
    name: str,
    module: nn.Module,
    target_modules: Optional[Iterable[str]] = None,
    square_only: bool = True,
) -> bool:
    """
    Decide whether one module should receive a LoRA branch.

    Input:
        name: module name from model.named_modules()
        module: nn.Module
        target_modules: optional name fragments.
        square_only: MiniMind original only attaches LoRA to square Linear layers.

    Output:
        bool
    """

    if not isinstance(name, str):
        raise ValueError("name must be a string.")

    if not isinstance(module, nn.Linear):
        return False

    if hasattr(module, "lora"):
        return False

    if target_modules is not None:
        targets = list(target_modules)
        if not any(target in name for target in targets):
            return False

    if square_only:
        out_features, in_features = module.weight.shape
        if out_features != in_features:
            return False

    return True


def apply_lora(
    model: nn.Module,
    rank: int = 16,
    target_modules: Optional[Iterable[str]] = None,
    square_only: bool = True,
) -> nn.Module:
    """
    Attach LoRA branches to target Linear modules.

    Input:
        model: nn.Module
        rank: int
        target_modules: optional name fragments.
        square_only: bool

    Output:
        model: nn.Module
    """

    if not isinstance(model, nn.Module):
        raise ValueError("model must be an nn.Module.")
    
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError("rank must be a positive integer.")
    
    for name, module in model.named_modules():
        if not should_apply_lora(
            name=name,
            module=module,
            target_modules=target_modules,
            square_only=square_only,
        ):
            continue

        out_features, in_features = module.weight.shape
        lora = LoRA(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
        ).to(device=module.weight.device, dtype=module.weight.dtype)

        setattr(module, "lora", lora)

        original_forward = module.forward

        # y = x @ W.T + LoRA(x)
        def forward_with_lora(
            x: torch.Tensor,
            layer_forward=original_forward,
            lora_layer=lora,
        ) -> torch.Tensor:
            return layer_forward(x) + lora_layer(x)
        
        module.forward = forward_with_lora

    return model


def iter_lora_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Return all modules that already have a `.lora` branch.

    Output:
        list[(module_name, module)]
    """

    if not isinstance(model, nn.Module):
        raise ValueError("model must be an nn.Module.")

    lora_modules = []

    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            lora_modules.append((name, module))

    return lora_modules


def iter_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Return trainable LoRA parameters.

    Output:
        list[nn.Parameter]
    """

    if not isinstance(model, nn.Module):
        raise ValueError("model must be an nn.Module")
    
    lora_params = []

    for _, module in iter_lora_modules(model):
        lora_params.extend(list(module.lora.parameters()))

    return lora_params


def mark_only_lora_as_trainable(model: nn.Module) -> List[nn.Parameter]:
    """
    Freeze base model parameters and leave only LoRA parameters trainable.

    Output:
        lora_params: list[nn.Parameter]
    """

    if not isinstance(model, nn.Module):
        raise ValueError("model must be an nn.Module.")

    for param in model.parameters():
        param.requires_grad = False

    lora_params = iter_lora_parameters(model)

    for param in lora_params:
        param.requires_grad = True

    return lora_params


def save_lora(model: nn.Module, path: str | Path) -> None:
    """
    Save only LoRA weights.

    Output file:
        dict[str, torch.Tensor]
            keys look like: "<module_name>.lora.<A_or_B_weight>"
    """

    if not isinstance(model, nn.Module):
        raise ValueError("model must be an nn.Module.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    raw_model = getattr(model, "_orig_mod", model)

    state_dict = {}

    for name, module in iter_lora_modules(raw_model):
        clean_name = name[7:] if name.startswith("module.") else name

        for key, value in module.lora.state_dict().items():
            state_dict[f"{clean_name}.lora.{key}"] = value.detach().cpu().half()

    torch.save(state_dict, path)


def load_lora(
    model: nn.Module,
    path: str | Path,
    device: Optional[str | torch.device] = None,
) -> None:
    """
    Load LoRA weights into a model that has already applied LoRA modules.
    """

    if not isinstance(model, nn.Module):
        raise ValueError("model must be an nn.Module.")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LoRA weight file not found: {path}")

    if device is None:
        device = next(model.parameters()).device

    state_dict = torch.load(path, map_location=device)
    state_dict = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }

    raw_model = getattr(model, "_orig_mod", model)

    for name, module in iter_lora_modules(raw_model):
        prefix = f"{name}.lora."

        lora_state = {
            key.replace(prefix, ""): value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }

        if lora_state:
            module.lora.load_state_dict(lora_state)


def merge_lora(
    model: nn.Module,
    lora_path: str | Path,
    save_path: str | Path,
    device: Optional[str | torch.device] = None,
) -> None:
    """
    Merge LoRA delta into base Linear weights and save one merged model state_dict.
    """

    if not isinstance(model, nn.Module):
        raise ValueError("model must be an nn.Module.")

    if device is None:
        device = next(model.parameters()).device

    load_lora(model, lora_path, device=device)

    raw_model = getattr(model, "_orig_mod", model)
    merged_state_dict = {}

    for key, value in raw_model.state_dict().items():
        if ".lora." not in key:
            merged_state_dict[key] = value.detach().cpu().half()

    for name, module in raw_model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        weight_key = f"{name}.weight"

        if weight_key not in merged_state_dict:
            continue

        if hasattr(module, "lora"):
            delta_weight = module.lora.B.weight.data @ module.lora.A.weight.data
            # y = x @ W + x @ Delta_W (Delta_W = B @ A)
            merged_weight = module.weight.data + delta_weight
            merged_state_dict[weight_key] = merged_weight.detach().cpu().half()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_state_dict, save_path)
