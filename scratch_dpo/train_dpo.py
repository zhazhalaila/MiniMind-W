from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model_minimind import MiniMindConfig
from scratch_dpo.dataloader import collate_dpo_batch
from scratch_dpo.dataset import DPODataset
from scratch_dpo.train_loop import compute_dpo_train_loss, move_dpo_batch_to_device
from scratch_pretrain.entry import build_model, load_checkpoint_file
from scratch_pretrain.optim import build_optimizer
from scratch_pretrain.tokenizer_utils import load_tokenizer
from scratch_pretrain.train_pretrain import (
    _build_formal_checkpoint_state,
    _save_resume_checkpoint,
    append_train_metric,
    build_autocast_context,
    build_grad_scaler,
    compute_learning_rate,
    format_train_log,
    set_optimizer_learning_rate,
)


def build_dpo_parser() -> argparse.ArgumentParser:
    """
    Build the formal DPO argument parser, aligned with MiniMind train_dpo.py.

    Expected argument groups:
        - path / saving
        - loading / resume
        - training
        - model
        - MoE
        - DPO beta

    Output:
        parser: argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser("formal_dpo")

    # path
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer")
    parser.add_argument("--data_path", type=str, default="data/dpo.jsonl")
    parser.add_argument("--log_dir", type=str, default="logs/dpo")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/dpo")
    parser.add_argument("--out_dir", type=str, default="out/dpo")

    # naming / loading
    parser.add_argument("--save_weight", type=str, default="dpo")
    parser.add_argument("--from_weight", type=str, default="out/full_sft_dense/full_sft_768_final.pt")
    parser.add_argument("--from_resume", type=str, default="none")

    # train
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=4e-8)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.15)

    # model
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=2)
    parser.add_argument("--intermediate_size", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--max_position_embeddings", type=int, default=32768)
    parser.add_argument("--rope_theta", type=float, default=1000000.0)
    parser.add_argument("--rms_norm_eps", type=float, default=1e-6)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--num_experts_per_tok", type=int, default=1)
    parser.add_argument("--moe_intermediate_size", type=int, default=None)
    parser.add_argument("--router_aux_loss_coef", type=float, default=5e-4)

    # data
    parser.add_argument("--empty_think_ratio", type=float, default=0.2)

    return parser


def parse_dpo_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for formal DPO training.

    Output:
        args: argparse.Namespace
    """

    parser = build_dpo_parser()
    return parser.parse_args(argv)


def build_model_config_from_args(args: argparse.Namespace) -> MiniMindConfig:
    """
    Convert DPO command-line args into MiniMindConfig.

    Output:
        model_config: MiniMindConfig
    """

    moe_intermediate_size = args.moe_intermediate_size
    if moe_intermediate_size is None:
        moe_intermediate_size = args.intermediate_size

    return MiniMindConfig(
        vocab_size=getattr(args, "vocab_size", None) or 6400,
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
        moe_intermediate_size=moe_intermediate_size,
        router_aux_loss_coef=args.router_aux_loss_coef,
    )


def load_policy_and_reference_models(
    weight_path: str,
    model_config: MiniMindConfig,
    device: str,
) -> Tuple[nn.Module, nn.Module]:
    """
    Load trainable policy model and frozen reference model from the same SFT weight.

    Input:
        weight_path: str
        model_config: MiniMindConfig
        device: str

    Output:
        policy_model: nn.Module
        ref_model: nn.Module
    """

    policy_model = build_model(
        model_config=model_config,
        device=device,
    )

    ref_model = build_model(
        model_config=model_config,
        device=device,
    )

    payload = load_checkpoint_file(
        checkpoint_path=weight_path,
        device=device,
    )

    if isinstance(payload, dict) and "model" in payload:
        state_dict = payload["model"]
    else:
        state_dict = payload

    policy_model.load_state_dict(state_dict, strict=True)
    ref_model.load_state_dict(state_dict, strict=True)

    policy_model.train()
    ref_model.eval()

    for param in ref_model.parameters():
        param.requires_grad_(False)

    return policy_model, ref_model


def run_formal_dpo(args: argparse.Namespace) -> List[float]:
    """
    Run formal DPO training.

    Output:
        loss_history: list[float]
    """

    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive.")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0.0 and 1.0.")
    if args.from_weight == "none":
        raise ValueError("DPO requires --from_weight to load the SFT policy/ref weights.")

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer_dir)
    args.vocab_size = len(tokenizer)

    dataset = DPODataset(
        records=args.data_path,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        empty_think_ratio=args.empty_think_ratio,
    )

    model_config = build_model_config_from_args(args)

    policy_model, ref_model = load_policy_and_reference_models(
        weight_path=args.from_weight,
        model_config=model_config,
        device=args.device,
    )

    optimizer = build_optimizer(
        model=policy_model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scaler = build_grad_scaler(args.device, args.dtype)

    weight_name = args.save_weight
    if args.use_moe:
        weight_name = f"{args.save_weight}_{args.hidden_size}_moe"
    else:
        weight_name = f"{args.save_weight}_{args.hidden_size}"

    metrics_path = Path(args.log_dir) / f"{weight_name}_metrics.jsonl"
    text_log_path = Path(args.log_dir) / f"{weight_name}.log"

    batches_per_epoch = (len(dataset) + args.batch_size - 1) // args.batch_size
    total_update_steps = (
        (batches_per_epoch + args.accumulation_steps - 1)
        // args.accumulation_steps
        * args.epochs
    )

    update_step = 0
    resume_epoch = 0
    resume_batch_idx = 0

    if args.from_resume != "none":
        checkpoint = load_checkpoint_file(
            checkpoint_path=args.from_resume,
            device=args.device,
        )
        policy_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

        update_step = int(checkpoint["step"])
        resume_epoch = int(checkpoint.get("epoch", 0))
        resume_batch_idx = int(checkpoint.get("batch_in_epoch", 0))

    loss_history: List[float] = []
    accum_loss = 0.0
    accum_count = 0

    optimizer.zero_grad()

    for epoch in range(resume_epoch, args.epochs):
        generator = torch.Generator()
        generator.manual_seed(42 + epoch)

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_dpo_batch,
            pin_memory=str(args.device).startswith("cuda"),
            generator=generator,
        )

        start_batch_idx = resume_batch_idx if epoch == resume_epoch else 0

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx < start_batch_idx:
                continue

            if accum_count == 0:
                current_lr = compute_learning_rate(
                    current_step=update_step + 1,
                    total_steps=total_update_steps,
                    base_learning_rate=args.learning_rate,
                    warmup_steps=args.warmup_steps,
                    min_lr_ratio=args.min_lr_ratio,
                )
                set_optimizer_learning_rate(optimizer, current_lr)

            policy_model.train()
            ref_model.eval()

            batch = move_dpo_batch_to_device(batch, args.device)

            with build_autocast_context(args.device, args.dtype):
                losses = compute_dpo_train_loss(
                    policy_model=policy_model,
                    ref_model=ref_model,
                    batch=batch,
                    beta=args.beta,
                )
                loss = losses["loss"]
                loss_for_backward = loss / args.accumulation_steps

            scaler.scale(loss_for_backward).backward()

            accum_loss += float(loss.item())
            accum_count += 1

            if accum_count < args.accumulation_steps:
                continue

            if args.grad_clip is not None and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    policy_model.parameters(),
                    args.grad_clip,
                )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            update_step += 1
            avg_loss = accum_loss / accum_count
            loss_history.append(avg_loss)

            if args.log_interval > 0 and update_step % args.log_interval == 0:
                log_line = format_train_log(update_step, avg_loss, current_lr)
                print(log_line)

                with text_log_path.open("a", encoding="utf-8") as f:
                    f.write(log_line + "\n")

                append_train_metric(
                    metrics_path=str(metrics_path),
                    step=update_step,
                    loss=avg_loss,
                    learning_rate=current_lr,
                )

            if args.save_interval > 0 and update_step % args.save_interval == 0:
                checkpoint = _build_formal_checkpoint_state(
                    model=policy_model,
                    optimizer=optimizer,
                    step=update_step,
                    epoch=epoch,
                    batch_in_epoch=batch_idx + 1,
                    scaler=scaler,
                )
                _save_resume_checkpoint(
                    checkpoint_dir=args.checkpoint_dir,
                    save_weight=weight_name,
                    checkpoint=checkpoint,
                )

            accum_loss = 0.0
            accum_count = 0

        resume_batch_idx = 0

    if accum_count > 0:
        if args.grad_clip is not None and args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                policy_model.parameters(),
                args.grad_clip,
            )

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        update_step += 1
        avg_loss = accum_loss / accum_count
        loss_history.append(avg_loss)

        log_line = format_train_log(update_step, avg_loss, current_lr)
        print(log_line)

        with text_log_path.open("a", encoding="utf-8") as f:
            f.write(log_line + "\n")

        append_train_metric(
            metrics_path=str(metrics_path),
            step=update_step,
            loss=avg_loss,
            learning_rate=current_lr,
        )

    final_checkpoint = _build_formal_checkpoint_state(
        model=policy_model,
        optimizer=optimizer,
        step=update_step,
        epoch=args.epochs,
        batch_in_epoch=0,
        scaler=scaler,
    )
    _save_resume_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        save_weight=weight_name,
        checkpoint=final_checkpoint,
    )

    final_weight_path = Path(args.out_dir) / f"{weight_name}_final.pt"
    torch.save(policy_model.state_dict(), final_weight_path)

    return loss_history


def main(argv: Optional[List[str]] = None) -> None:
    """
    Formal executable entry for DPO.
    """

    args = parse_dpo_args(argv)
    run_formal_dpo(args)


if __name__ == "__main__":
    main()
