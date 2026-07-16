from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import torch

from model.model_minimind import MiniMindConfig
from scratch_lora.config import (
    LoRADataConfig,
    LoRATrainConfig,
    build_lora_data_config,
    build_lora_train_config,
)
from scratch_lora.lora import (
    apply_lora,
    iter_lora_modules,
    mark_only_lora_as_trainable,
    parse_target_modules,
    save_lora,
)
from scratch_lora.train_loop import run_lora_train_loop
from scratch_pretrain.entry import build_model
from scratch_pretrain.tokenizer_utils import load_tokenizer
from scratch_sft.dataloader import build_sft_dataloader
from scratch_sft.dataset import SFTDataset


def build_lora_runtime(
    data_config: LoRADataConfig,
    train_config: LoRATrainConfig,
    model_config: MiniMindConfig,
) -> Dict[str, object]:
    """
    Build tokenizer, dataset, dataloader, base model with LoRA, and optimizer.

    Output:
        runtime: dict[str, object]
            expected keys:
            - "tokenizer"
            - "dataset"
            - "dataloader"
            - "model"
            - "optimizer"
            - "lora_params"
    """

    tokenizer = load_tokenizer(data_config.tokenizer_dir)

    dataset = SFTDataset(
        data_path=data_config.data_path,
        tokenizer=tokenizer,
        max_seq_len=data_config.max_seq_len,
        add_system_ratio=data_config.add_system_ratio,
        empty_think_ratio=data_config.empty_think_ratio,
    )

    dataloader = build_sft_dataloader(
        dataset=dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
    )

    model = build_model(
        model_config=model_config,
        device=train_config.device,
    )

    if train_config.from_weight != "none":
        payload = torch.load(train_config.from_weight, map_location=train_config.device)
        state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        model.load_state_dict(state_dict)

    target_modules = parse_target_modules(train_config.target_modules)
    apply_lora(
        model=model,
        rank=train_config.rank,
        target_modules=target_modules,
        square_only=True,
    )

    lora_params = mark_only_lora_as_trainable(model)
    if not lora_params:
        raise ValueError("No LoRA parameters were attached to the model.")

    optimizer = torch.optim.AdamW(
        lora_params,
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    if train_config.from_resume != "none":
        checkpoint = torch.load(train_config.from_resume, map_location=train_config.device)
        lora_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

        for name, module in iter_lora_modules(model):
            prefix = f"{name}.lora."
            module_state = {
                key.replace(prefix, ""): value
                for key, value in lora_state.items()
                if key.startswith(prefix)
            }
            if module_state:
                module.lora.load_state_dict(module_state)

        if isinstance(checkpoint, dict) and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

    return {
        "tokenizer": tokenizer,
        "dataset": dataset,
        "dataloader": dataloader,
        "model": model,
        "optimizer": optimizer,
        "lora_params": lora_params,
    }


def build_lora_smoke_test_configs(
    project_root: str,
    device: str = "cpu",
) -> Tuple[LoRADataConfig, LoRATrainConfig, MiniMindConfig]:
    """
    Build tiny configs for local LoRA smoke test.

    Output:
        data_config: LoRADataConfig
        train_config: LoRATrainConfig
        model_config: MiniMindConfig
    """

    project_root_path = Path(project_root)
    smoke_data_path = project_root_path / "data" / "lora_smoke.jsonl"
    smoke_data_path.parent.mkdir(parents=True, exist_ok=True)

    if not smoke_data_path.exists():
        smoke_record = {
            "conversations": [
                {"role": "user", "content": "你是谁？"},
                {"role": "assistant", "content": "我是 MiniMind-W 的 LoRA smoke test 助手。"},
            ]
        }
        with smoke_data_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(smoke_record, ensure_ascii=False) + "\n")

    data_config = build_lora_data_config(
        tokenizer_dir=str(project_root_path / "tokenizer"),
        data_path=str(smoke_data_path),
        max_seq_len=64,
        add_system_ratio=0.0,
        empty_think_ratio=0.0,
    )

    train_config = build_lora_train_config(
        log_dir=str(project_root_path / "logs" / "lora_smoke"),
        checkpoint_dir=str(project_root_path / "checkpoints" / "lora_smoke"),
        out_dir=str(project_root_path / "out" / "lora_smoke"),
        lora_name="lora_smoke",
        from_weight="none",
        from_resume="none",
        epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        device=device,
        dtype="float32",
        num_workers=0,
        accumulation_steps=1,
        grad_clip=1.0,
        log_interval=1,
        save_interval=1,
        warmup_steps=0,
        min_lr_ratio=0.1,
        rank=2,
        target_modules=None,
    )

    model_config = MiniMindConfig(
        vocab_size=6400,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )

    return data_config, train_config, model_config


def run_lora_smoke_test(
    project_root: str,
    device: str = "cpu",
) -> list[float]:
    """
    Run one tiny local LoRA smoke test.

    Output:
        loss_history: list[float]
    """

    data_config, train_config, model_config = build_lora_smoke_test_configs(
        project_root=project_root,
        device=device,
    )

    runtime = build_lora_runtime(
        data_config=data_config,
        train_config=train_config,
        model_config=model_config,
    )

    loss_history = run_lora_train_loop(
        model=runtime["model"],
        dataloader=runtime["dataloader"],
        optimizer=runtime["optimizer"],
        device=train_config.device,
        max_steps=1,
        log_every=train_config.log_interval,
        save_every=train_config.save_interval,
        save_dir=train_config.checkpoint_dir,
    )

    save_lora(
        runtime["model"],
        Path(train_config.out_dir) / f"{train_config.lora_name}_{model_config.hidden_size}.pt",
    )

    return loss_history


def main() -> None:
    run_lora_smoke_test(project_root=".", device="cpu")


if __name__ == "__main__":
    main()
