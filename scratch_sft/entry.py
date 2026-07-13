from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model_minimind import MiniMindConfig
from scratch_pretrain.entry import build_model
from scratch_pretrain.optim import build_optimizer
from scratch_pretrain.tokenizer_utils import load_tokenizer
from scratch_sft.config import SFTDataConfig, SFTTrainConfig, build_sft_data_config, build_sft_train_config
from scratch_sft.dataloader import build_sft_dataloader
from scratch_sft.dataset import SFTDataset
from scratch_sft.train_loop import run_sft_train_loop


def build_sft_runtime(
    data_config: SFTDataConfig,
    train_config: SFTTrainConfig,
    model_config: MiniMindConfig,
) -> Dict[str, Any]:
    """
    Assemble tokenizer, dataset, dataloader, model, and optimizer for SFT.

    Output:
        runtime["tokenizer"]
        runtime["dataset"]
        runtime["dataloader"]
        runtime["model"]
        runtime["optimizer"]
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

    optimizer = build_optimizer(
        model=model,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    return {
        "tokenizer": tokenizer,
        "dataset": dataset,
        "dataloader": dataloader,
        "model": model,
        "optimizer": optimizer,
    }


def run_sft_entry(
    data_config: SFTDataConfig,
    train_config: SFTTrainConfig,
    model_config: MiniMindConfig,
) -> list[float]:
    """
    Run one minimal SFT entry point.

    Output:
        loss_history: list[float]
    """

    runtime = build_sft_runtime(
        data_config=data_config,
        train_config=train_config,
        model_config=model_config,
    )

    return run_sft_train_loop(
        model=runtime["model"],
        dataloader=runtime["dataloader"],
        optimizer=runtime["optimizer"],
        device=train_config.device,
        max_steps=train_config.max_steps,
        log_every=train_config.log_every,
        save_every=train_config.save_every,
        save_dir=train_config.save_dir,
    )


def build_sft_smoke_test_configs(
    project_root: str,
    device: str = "cpu",
) -> Tuple[SFTDataConfig, SFTTrainConfig, MiniMindConfig]:
    """
    Build one tiny smoke-test config triplet for SFT.

    Input:
        project_root: str
        device: str

    Output:
        data_config: SFTDataConfig
        train_config: SFTTrainConfig
        model_config: MiniMindConfig
    """

    project_root_path = Path(project_root)

    data_config = build_sft_data_config(
        tokenizer_dir=str(project_root_path / "tokenizer"),
        data_path=str(project_root_path / "data" / "sft_t2t_mini.jsonl"),
        max_seq_len=64,
        add_system_ratio=0.0,
        empty_think_ratio=0.0,
    )

    train_config = build_sft_train_config(
        save_dir=str(project_root_path / "checkpoints_smoke"),
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.01,
        max_steps=2,
        device=device,
        log_every=1,
        save_every=2,
    )

    model_config = MiniMindConfig(
        vocab_size=6400,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )

    return data_config, train_config, model_config


def run_sft_smoke_test(
    project_root: str,
    device: str = "cpu",
) -> list[float]:
    """
    Run one tiny local SFT smoke test.

    Output:
        loss_history: list[float]
    """

    data_config, train_config, model_config = build_sft_smoke_test_configs(
        project_root=project_root,
        device=device,
    )

    return run_sft_entry(
        data_config=data_config,
        train_config=train_config,
        model_config=model_config,
    )


def main() -> None:
    """
    Minimal executable entry for local SFT smoke testing.
    """

    project_root = str(Path(__file__).resolve().parents[1])
    run_sft_smoke_test(project_root=project_root, device="cpu")


if __name__ == "__main__":
    main()