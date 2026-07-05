from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import torch

from pathlib import Path

from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from scratch_pretrain.config import (
    PretrainDataConfig,
    PretrainTrainConfig,
    build_pretrain_data_config,
    build_pretrain_train_config,
)
from scratch_pretrain.dataloader import build_pretrain_dataloader
from scratch_pretrain.optim import build_optimizer
from scratch_pretrain.tokenizer_utils import load_tokenizer
from scratch_pretrain.train_loop import run_pretrain_train_loop


def build_model(
    model_config: MiniMindConfig,
    device: Union[str, torch.device],
) -> MiniMindForCausalLM:
    """
    Build the causal LM model and move it onto the target device.

    Input:
        model_config: MiniMindConfig
        device: str | torch.device

    Output:
        model: MiniMindForCausalLM
    """

    model = MiniMindForCausalLM(model_config)
    model = model.to(device)
    return model


def build_pretrain_runtime(
    data_config: PretrainDataConfig,
    train_config: PretrainTrainConfig,
    model_config: MiniMindConfig,
) -> Dict[str, Any]:
    """
    Build the runtime objects required by the pretrain entry.

    Input:
        data_config: PretrainDataConfig
        train_config: PretrainTrainConfig
        model_config: MiniMindConfig

    Output:
        runtime: dict[str, Any]
            Expected keys:
            - "tokenizer"
            - "dataset"
            - "dataloader"
            - "model"
            - "optimizer"
    """

    # convert dataset into tokens
    tokenizer = load_tokenizer(data_config.tokenizer_dir)

    dataset = PretrainDataset(
        data_path=data_config.data_path,
        tokenizer=tokenizer,
        max_length=data_config.max_length,
    )

    dataloader = build_pretrain_dataloader(
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

    runtime = {
        "tokenizer": tokenizer,
        "dataset": dataset,
        "dataloader": dataloader,
        "model": model,
        "optimizer": optimizer,
    }

    return runtime

def run_pretrain_entry(
    data_config: PretrainDataConfig,
    train_config: PretrainTrainConfig,
    model_config: MiniMindConfig,
) -> List[float]:
    """
    Run the minimal pretrain entry.

    Input:
        data_config: PretrainDataConfig
        train_config: PretrainTrainConfig
        model_config: MiniMindConfig

    Output:
        loss_history: list[float]
            Expected length: train_config.max_steps
    """

    runtime = build_pretrain_runtime(
        data_config=data_config,
        train_config=train_config,
        model_config=model_config,
    )

    loss_history = run_pretrain_train_loop(
        model=runtime["model"],
        dataloader=runtime["dataloader"],
        optimizer=runtime["optimizer"],
        device=train_config.device,
        max_steps=train_config.max_steps,
        log_every=train_config.log_every,
        save_every=train_config.save_every,
        save_dir=train_config.save_dir,
    )

    return loss_history


def load_checkpoint_file(
    checkpoint_path: str,
    device: Union[str, torch.device],
) -> Dict[str, Any]:
    """
    Load one checkpoint from local disk.

    Input:
        checkpoint_path: str
        device: str | torch.device

    Output:
        checkpoint: dict[str, Any]
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    return checkpoint


def build_smoke_test_configs(
    project_root: str,
    device: str = "cpu",
) -> Tuple[PretrainDataConfig, PretrainTrainConfig, MiniMindConfig]:
    """
    Build minimal configs for the day-4 smoke test.

    Input:
        project_root: str
        device: str

    Output:
        data_config: PretrainDataConfig
        train_config: PretrainTrainConfig
        model_config: MiniMindConfig

    Suggested behavior:
        - use project_root/data/pretrain_t2t_mini.jsonl
        - use project_root/tokenizer
        - use a small save_dir under project_root
        - use a tiny model config and a tiny max_steps
    """
    project_root = Path(project_root)

    data_config = build_pretrain_data_config(
        tokenizer_dir=str(project_root / "tokenizer"),
        data_path=str(project_root / "data" / "pretrain_t2t_mini.jsonl"),
        max_length=8,
    )

    train_config = build_pretrain_train_config(
        save_dir=str(project_root / "checkpoints_smoke"),
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
        max_position_embeddings=32,
    )

    return data_config, train_config, model_config


def run_pretrain_smoke_test(
    project_root: str,
    device: str = "cpu",
) -> List[float]:
    """
    Run the day-4 minimal smoke test.

    Input:
        project_root: str
        device: str

    Output:
        loss_history: list[float]
    """
    data_config, train_config, model_config = build_smoke_test_configs(
        project_root=project_root,
        device=device,
    )

    loss_history = run_pretrain_entry(
        data_config=data_config,
        train_config=train_config,
        model_config=model_config,
    )

    return loss_history

def main() -> None:
    """
    Minimal executable entry for the day-4 smoke test.
    """

    project_root = str(Path(__file__).resolve().parents[1])
    run_pretrain_smoke_test(project_root=project_root, device="cpu")
