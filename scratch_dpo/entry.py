from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.model_minimind import MiniMindConfig
from scratch_dpo.config import DPODataConfig, DPOTrainConfig
from scratch_dpo.dataloader import build_dpo_dataloader
from scratch_dpo.dataset import DPODataset
from scratch_dpo.train_dpo import load_policy_and_reference_models
from scratch_dpo.train_loop import run_dpo_train_loop
from scratch_pretrain.entry import build_model
from scratch_pretrain.optim import build_optimizer
from scratch_pretrain.tokenizer_utils import load_tokenizer


def build_dpo_runtime(
    data_config: DPODataConfig,
    train_config: DPOTrainConfig,
    model_config: MiniMindConfig,
) -> Dict[str, Any]:
    """
    Assemble tokenizer, dataset, dataloader, policy model, reference model, and optimizer.

    Output:
        runtime: dict[str, Any]
            Expected keys:
            - "tokenizer"
            - "dataset"
            - "dataloader"
            - "policy_model"
            - "ref_model"
            - "optimizer"
    """

    tokenizer = load_tokenizer(data_config.tokenizer_dir)

    dataset = DPODataset(
        records=data_config.data_path,
        tokenizer=tokenizer,
        max_seq_len=data_config.max_seq_len,
        empty_think_ratio=data_config.empty_think_ratio,
    )

    dataloader = build_dpo_dataloader(
        dataset=dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
    )

    policy_model, ref_model = load_policy_and_reference_models(
        weight_path=train_config.from_weight,
        model_config=model_config,
        device=train_config.device,
    )

    optimizer = build_optimizer(
        model=policy_model,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    return {
        "tokenizer": tokenizer,
        "dataset": dataset,
        "dataloader": dataloader,
        "policy_model": policy_model,
        "ref_model": ref_model,
        "optimizer": optimizer,
    }


def build_dpo_smoke_test_configs(
    project_root: str,
    device: str = "cpu",
) -> Tuple[DPODataConfig, DPOTrainConfig, MiniMindConfig]:
    """
    Build tiny configs for a local DPO smoke test.

    Output:
        data_config: DPODataConfig
        train_config: DPOTrainConfig
        model_config: MiniMindConfig
    """

    tmp_root = Path(project_root) / ".tmp_dpo_smoke"

    data_config = DPODataConfig(
        tokenizer_dir=str(Path(project_root) / "tokenizer"),
        data_path=str(tmp_root / "toy_dpo.jsonl"),
        max_seq_len=32,
        empty_think_ratio=0.0,
    )

    train_config = DPOTrainConfig(
        log_dir=str(tmp_root / "logs"),
        checkpoint_dir=str(tmp_root / "checkpoints"),
        out_dir=str(tmp_root / "out"),
        save_weight="dpo_smoke",
        from_weight=str(tmp_root / "sft_seed.pt"),
        from_resume="none",
        epochs=1,
        batch_size=1,
        learning_rate=1e-4,
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
        beta=0.15,
    )

    model_config = MiniMindConfig(
        vocab_size=6400,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )

    return data_config, train_config, model_config


def run_dpo_smoke_test(
    project_root: str,
    device: str = "cpu",
) -> list[float]:
    """
    Run one tiny local DPO smoke test.

    Output:
        loss_history: list[float]
    """

    data_config, train_config, model_config = build_dpo_smoke_test_configs(
        project_root=project_root,
        device=device,
    )

    tmp_root = Path(train_config.from_weight).parent
    tmp_root.mkdir(parents=True, exist_ok=True)
    Path(train_config.log_dir).mkdir(parents=True, exist_ok=True)
    Path(train_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(train_config.out_dir).mkdir(parents=True, exist_ok=True)

    toy_record = (
        '{"chosen":[{"role":"user","content":"hi"},'
        '{"role":"assistant","content":"hello"}],'
        '"rejected":[{"role":"user","content":"hi"},'
        '{"role":"assistant","content":"bad"}]}\n'
    )
    Path(data_config.data_path).write_text(toy_record, encoding="utf-8")

    tokenizer = load_tokenizer(data_config.tokenizer_dir)
    model_config = MiniMindConfig(
        vocab_size=len(tokenizer),
        hidden_size=model_config.hidden_size,
        intermediate_size=model_config.intermediate_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        num_key_value_heads=model_config.num_key_value_heads,
        max_position_embeddings=model_config.max_position_embeddings,
    )

    seed_model = build_model(
        model_config=model_config,
        device=device,
    )
    torch.save(seed_model.state_dict(), train_config.from_weight)

    runtime = build_dpo_runtime(
        data_config=data_config,
        train_config=train_config,
        model_config=model_config,
    )

    return run_dpo_train_loop(
        policy_model=runtime["policy_model"],
        ref_model=runtime["ref_model"],
        dataloader=runtime["dataloader"],
        optimizer=runtime["optimizer"],
        device=train_config.device,
        max_steps=1,
        beta=train_config.beta,
        log_every=train_config.log_interval,
        save_every=None,
        save_dir=None,
    )


def main() -> None:
    """
    Minimal executable entry for local DPO smoke testing.
    """

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    run_dpo_smoke_test(project_root=project_root, device="cpu")
