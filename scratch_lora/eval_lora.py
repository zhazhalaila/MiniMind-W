from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple, Union

import torch
from torch import nn

from model.model_minimind import MiniMindConfig
from scratch_lora.lora import apply_lora, load_lora
from scratch_pretrain.entry import build_model
from scratch_pretrain.tokenizer_utils import load_tokenizer


def load_lora_inference_artifacts(
    base_weight_path: str,
    lora_weight_path: str,
    tokenizer_dir: str,
    model_config: MiniMindConfig,
    device: Union[str, torch.device],
) -> Tuple[Any, nn.Module]:
    """
    Load tokenizer, base model weight and LoRA weight for inference.

    Input:
        base_weight_path: str
        lora_weight_path: str
        tokenizer_dir: str
        model_config: MiniMindConfig
        device: str | torch.device

    Output:
        tokenizer: Any
        model: nn.Module
    """

    base_weight_path = Path(base_weight_path)
    lora_weight_path = Path(lora_weight_path)

    if not base_weight_path.exists():
        raise FileNotFoundError(f"base weight file not found: {base_weight_path}")

    if not lora_weight_path.exists():
        raise FileNotFoundError(f"LoRA weight file not found: {lora_weight_path}")

    tokenizer = load_tokenizer(tokenizer_dir)

    model = build_model(
        model_config=model_config,
        device=device,
    )

    payload = torch.load(base_weight_path, map_location=device)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state_dict)

    lora_state = torch.load(lora_weight_path, map_location=device)
    rank = 16
    for key, value in lora_state.items():
        if key.endswith(".lora.A.weight"):
            rank = int(value.shape[0])
            break

    apply_lora(model, rank=rank)
    load_lora(model, lora_weight_path, device=device)

    model.eval()
    return tokenizer, model
