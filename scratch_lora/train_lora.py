from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model_minimind import MiniMindConfig
from pathlib import Path

from scratch_lora.config import build_lora_data_config, build_lora_train_config
from scratch_lora.entry import build_lora_runtime
from scratch_lora.lora import save_lora
from scratch_lora.train_loop import run_lora_train_loop
from scratch_pretrain.tokenizer_utils import load_tokenizer


def build_lora_parser() -> argparse.ArgumentParser:
    """
    Build the formal LoRA training parser, aligned with MiniMind train_lora.py.

    Output:
        parser: argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser("formal_lora")

    # path
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer")
    parser.add_argument("--data_path", type=str, default="data/lora_medical.jsonl")
    parser.add_argument("--log_dir", type=str, default="logs/lora")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/lora")
    parser.add_argument("--out_dir", type=str, default="out/lora")

    # naming / loading
    parser.add_argument("--lora_name", type=str, default="lora_medical")
    parser.add_argument("--from_weight", type=str, default="out/full_sft_dense/full_sft_768_final.pt")
    parser.add_argument("--from_resume", type=str, default="none")

    # train
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)

    # data
    parser.add_argument("--max_seq_len", type=int, default=340)
    parser.add_argument("--add_system_ratio", type=float, default=0.2)
    parser.add_argument("--empty_think_ratio", type=float, default=0.2)

    # model
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=2)
    parser.add_argument("--intermediate_size", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--max_position_embeddings", type=int, default=32768)
    parser.add_argument("--rope_theta", type=float, default=1000000.0)
    parser.add_argument("--rms_norm_eps", type=float, default=1e-6)

    # MoE
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--num_experts_per_tok", type=int, default=1)
    parser.add_argument("--moe_intermediate_size", type=int, default=2432)
    parser.add_argument("--router_aux_loss_coef", type=float, default=5e-4)

    # LoRA
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--target_modules", type=str, default=None)

    return parser


def parse_lora_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for formal LoRA training.
    """

    parser = build_lora_parser()
    return parser.parse_args(argv)


def build_model_config_from_args(args: argparse.Namespace) -> MiniMindConfig:
    """
    Build MiniMindConfig from LoRA CLI args.

    Output:
        MiniMindConfig
    """

    vocab_size = args.vocab_size
    if vocab_size is None:
        raise ValueError(
            "vocab_size is None. Set args.vocab_size before calling build_model_config_from_args."
        )

    return MiniMindConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        max_position_embeddings=args.max_position_embeddings,
        rope_theta=args.rope_theta,
        rms_norm_eps=args.rms_norm_eps,
        use_moe=bool(args.use_moe),
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        moe_intermediate_size=args.moe_intermediate_size,
        router_aux_loss_coef=args.router_aux_loss_coef,
    )


def run_formal_lora(args: argparse.Namespace) -> List[float]:
    """
    Formal LoRA training entry.

    Output:
        loss_history: list[float]
    """

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer_dir)

    if args.vocab_size is None:
        args.vocab_size = len(tokenizer)

    data_config = build_lora_data_config(
        tokenizer_dir=args.tokenizer_dir,
        data_path=args.data_path,
        max_seq_len=args.max_seq_len,
        add_system_ratio=args.add_system_ratio,
        empty_think_ratio=args.empty_think_ratio,
    )

    train_config = build_lora_train_config(
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        out_dir=args.out_dir,
        lora_name=args.lora_name,
        from_weight=args.from_weight,
        from_resume=args.from_resume,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        dtype=args.dtype,
        num_workers=args.num_workers,
        accumulation_steps=args.accumulation_steps,
        grad_clip=args.grad_clip,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        rank=args.rank,
        target_modules=args.target_modules,
    )

    model_config = build_model_config_from_args(args)

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
        max_steps=args.epochs * len(runtime["dataloader"]),
        log_every=train_config.log_interval,
        save_every=train_config.save_interval,
        save_dir=train_config.checkpoint_dir,
    )

    lora_path = Path(train_config.out_dir) / f"{train_config.lora_name}_{model_config.hidden_size}.pt"
    save_lora(runtime["model"], lora_path)

    return loss_history


def main(argv: Optional[List[str]] = None) -> None:
    """
    Command-line entry.
    """

    args = parse_lora_args(argv)
    run_formal_lora(args)


if __name__ == "__main__":
    main()

