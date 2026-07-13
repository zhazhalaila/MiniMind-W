from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model_minimind import MiniMindConfig
from scratch_pretrain.entry import build_model, load_checkpoint_file
from scratch_pretrain.moe import (
    build_moe_kwargs_from_args,
    build_moe_weight_name,
    collect_router_aux_loss,
    combine_lm_and_router_loss,
)
from scratch_pretrain.optim import build_optimizer
from scratch_pretrain.tokenizer_utils import load_tokenizer
from scratch_pretrain.train_pretrain import (
    _build_formal_checkpoint_state,
    _normalize_resume_position,
    _save_resume_checkpoint,
    append_train_metric,
    build_autocast_context,
    build_grad_scaler,
    compute_learning_rate,
    format_train_log,
    set_optimizer_learning_rate,
)
from scratch_sft.dataloader import collate_sft_batch
from scratch_sft.dataset import SFTDataset


def build_sft_parser() -> argparse.ArgumentParser:
    """
    Build the formal SFT argument parser.

    Suggested argument groups:
        - path
        - load / save
        - train
        - model
    """

    parser = argparse.ArgumentParser("formal_sft")

    # path
    parser.add_argument("--tokenizer_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--out_dir", type=str, default="out")

    # naming / loading
    parser.add_argument("--save_weight", type=str, default="full_sft")
    parser.add_argument("--from_weight", type=str, default="none")
    parser.add_argument("--from_resume", type=str, default="none")

    # train
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)

    # SFT data
    parser.add_argument("--max_seq_len", type=int, default=768)
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

    return parser


def parse_sft_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the formal SFT script.
    """

    parser = build_sft_parser()
    return parser.parse_args(argv)


def build_model_config_from_args(args: argparse.Namespace) -> MiniMindConfig:
    """
    Convert command-line args into one MiniMindConfig for SFT.

    Output:
        model_config: MiniMindConfig
    """

    if not isinstance(args, argparse.Namespace):
        raise TypeError("args must be an argparse.Namespace.")

    moe_kwargs = build_moe_kwargs_from_args(args)

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
        **moe_kwargs,
    )


def run_formal_sft(args: argparse.Namespace) -> list[float]:
    """
    Run the formal SFT training entry.

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
    if args.from_resume != "none" and args.from_weight != "none":
        raise ValueError("from_resume and from_weight cannot both be set.")

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer_dir)
    if getattr(args, "vocab_size", None) is None:
        args.vocab_size = len(tokenizer)

    model_config = build_model_config_from_args(args)
    dataset = SFTDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        add_system_ratio=args.add_system_ratio,
        empty_think_ratio=args.empty_think_ratio,
    )
    model = build_model(
        model_config=model_config,
        device=args.device,
    )
    optimizer = build_optimizer(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = build_grad_scaler(args.device, args.dtype)

    weight_name = build_moe_weight_name(
        save_weight=args.save_weight,
        hidden_size=args.hidden_size,
        use_moe=bool(args.use_moe),
    )
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
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        update_step = int(checkpoint["step"])
        resume_epoch = int(checkpoint.get("epoch", 0))
        resume_batch_idx = int(checkpoint.get("batch_in_epoch", 0))
    elif args.from_weight != "none":
        payload = load_checkpoint_file(
            checkpoint_path=args.from_weight,
            device=args.device,
        )
        if isinstance(payload, dict) and "model" in payload:
            model.load_state_dict(payload["model"])
        else:
            model.load_state_dict(payload)

    loss_history: list[float] = []
    accum_loss = 0.0
    accum_count = 0
    current_lr = optimizer.param_groups[0]["lr"]

    optimizer.zero_grad()

    for epoch in range(resume_epoch, args.epochs):
        generator = torch.Generator()
        generator.manual_seed(42 + epoch)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_sft_batch,
            pin_memory=args.device.startswith("cuda"),
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

            model.train()
            batch = {
                key: value.to(args.device)
                for key, value in batch.items()
            }

            with build_autocast_context(args.device, args.dtype):
                outputs = model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                )
                lm_loss = outputs.loss
                if bool(args.use_moe):
                    router_aux_loss = collect_router_aux_loss(model)
                    loss = combine_lm_and_router_loss(lm_loss, router_aux_loss)
                else:
                    loss = lm_loss

            loss_value = float(loss.item())
            loss_for_backward = loss / args.accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            accum_loss += loss_value
            accum_count += 1

            if accum_count < args.accumulation_steps:
                continue

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
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
                next_epoch, next_batch_in_epoch = _normalize_resume_position(
                    epoch=epoch,
                    batch_in_epoch=batch_idx + 1,
                    total_batches=len(dataloader),
                )
                _save_resume_checkpoint(
                    checkpoint_dir=args.checkpoint_dir,
                    save_weight=weight_name,
                    checkpoint=_build_formal_checkpoint_state(
                        model=model,
                        optimizer=optimizer,
                        step=update_step,
                        epoch=next_epoch,
                        batch_in_epoch=next_batch_in_epoch,
                        scaler=scaler,
                    ),
                )
                torch.save(
                    model.state_dict(),
                    Path(args.out_dir) / f"{weight_name}.pt",
                )

            accum_loss = 0.0
            accum_count = 0

        if accum_count > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
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

            accum_loss = 0.0
            accum_count = 0

        resume_batch_idx = 0

    _save_resume_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        save_weight=weight_name,
        checkpoint=_build_formal_checkpoint_state(
            model=model,
            optimizer=optimizer,
            step=update_step,
            epoch=args.epochs,
            batch_in_epoch=0,
            scaler=scaler,
        ),
    )
    torch.save(
        model.state_dict(),
        Path(args.out_dir) / f"{weight_name}_final.pt",
    )

    return loss_history


def main(argv: Optional[List[str]] = None) -> None:
    """
    Formal executable entry for SFT.
    """

    args = parse_sft_args(argv)
    run_formal_sft(args)


if __name__ == "__main__":
    main()
