from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig
from scratch_pretrain.config import (
    PretrainDataConfig,
    PretrainTrainConfig,
    build_pretrain_data_config,
    build_pretrain_train_config,
)
from scratch_pretrain.entry import build_model, load_checkpoint_file
from scratch_pretrain.dataloader import build_pretrain_dataloader
from scratch_pretrain.optim import build_optimizer
from scratch_pretrain.tokenizer_utils import load_tokenizer


def build_train_parser() -> argparse.ArgumentParser:
    """
    Build the formal pretrain argument parser.

    Suggested arguments:
        --log_dir
        --checkpoint_dir
        --out_dir
        --save_weight
        --epochs
        --batch_size
        --learning_rate
        --device
        --dtype
        --num_workers
        --accumulation_steps
        --grad_clip
        --log_interval
        --save_interval
        --hidden_size
        --num_hidden_layers
        --max_seq_len
        --tokenizer_dir
        --data_path
        --from_weight
        --from_resume
    """
    parser = argparse.ArgumentParser("formal_pretrain")

    # path
    parser.add_argument("--tokenizer_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--out_dir", type=str, default="out")

    # naming / loading
    parser.add_argument("--save_weight", type=str, default="pretrain")
    parser.add_argument("--from_weight", type=str, default="none")
    parser.add_argument("--from_resume", type=str, default="none")

    # train
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)

    # model
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--num_key_value_heads", type=int, default=4)
    parser.add_argument("--intermediate_size", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=128)

    return parser


def parse_train_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the formal pretrain script.
    """

    parser = build_train_parser()
    args = parser.parse_args(argv)

    if args.epochs <= 0:
        parser.error("--epochs must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive.")
    if args.accumulation_steps <= 0:
        parser.error("--accumulation_steps must be positive.")
    if args.warmup_steps < 0:
        parser.error("--warmup_steps must be non-negative.")
    if args.num_workers < 0:
        parser.error("--num_workers must be non-negative.")
    if args.log_interval < 0:
        parser.error("--log_interval must be non-negative.")
    if args.save_interval < 0:
        parser.error("--save_interval must be non-negative.")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        parser.error("--min_lr_ratio must be between 0.0 and 1.0.")
    if args.from_resume != "none" and args.from_weight != "none":
        parser.error("--from_resume and --from_weight cannot both be set.")

    return args


def build_data_config_from_args(args: argparse.Namespace) -> PretrainDataConfig:
    """
    Build the data config from parsed command-line arguments.

    Expected input fields:
        args.tokenizer_dir: str
        args.data_path: str
        args.max_seq_len: int

    Output:
        PretrainDataConfig
    """

    return build_pretrain_data_config(
        tokenizer_dir=args.tokenizer_dir,
        data_path=args.data_path,
        max_length=args.max_seq_len,
    )


def build_train_config_from_args(args: argparse.Namespace) -> PretrainTrainConfig:
    """
    Build the train config from parsed command-line arguments.

    Expected input fields:
        args.checkpoint_dir: str
        args.batch_size: int
        args.learning_rate: float
        args.epochs: int
        args.device: str
        args.log_interval: int
        args.save_interval: int

    Output:
        PretrainTrainConfig

    Notes:
        You may choose to map:
        - args.checkpoint_dir -> save_dir
        - args.epochs into a smaller training-control field for
          the minimal train loop used in MiniMind-W.
    """

    return build_pretrain_train_config(
        save_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.epochs,
        device=args.device,
        log_every=args.log_interval,
        save_every=args.save_interval,
    )


def build_model_config_from_args(args: argparse.Namespace) -> MiniMindConfig:
    """
    Build the model config from parsed command-line arguments.

    Expected input fields:
        args.hidden_size: int
        args.num_hidden_layers: int

    Output:
        MiniMindConfig
    """

    return MiniMindConfig(
        vocab_size=getattr(args, "vocab_size", 6400),
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        max_position_embeddings=args.max_seq_len,
    )


def build_runtime_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Build tokenizer / dataset / dataloader / model / optimizer from parsed args.

    Output:
        runtime: dict[str, Any]
            Expected keys:
            - "tokenizer"
            - "dataset"
            - "dataloader"
            - "model"
            - "optimizer"
    """

    data_config = build_data_config_from_args(args)
    train_config = build_train_config_from_args(args)

    tokenizer = load_tokenizer(data_config.tokenizer_dir)

    vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else getattr(tokenizer, "vocab_size", 6400)

    model_config = MiniMindConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        max_position_embeddings=args.max_seq_len,
    )

    dataset = PretrainDataset(
        data_path=data_config.data_path,
        tokenizer=tokenizer,
        max_length=data_config.max_length,
    )

    dataloader = build_pretrain_dataloader(
        dataset=dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
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


def build_autocast_context(
    device: str,
    dtype: str,
) -> Any:
    """
    Build the precision context used by the formal pretrain script.

    Input:
        device: str
        dtype: str

    Output:
        autocast_context: Any
    """

    if not device.startswith("cuda"):
        return nullcontext()

    if dtype == "float16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    if dtype == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    return nullcontext()


def build_grad_scaler(
    device: str,
    dtype: str,
) -> torch.cuda.amp.GradScaler:
    enabled = device.startswith("cuda") and dtype == "float16"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def compute_learning_rate(
    current_step: int,
    total_steps: int,
    base_learning_rate: float,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
) -> float:
    if total_steps <= 0:
        return base_learning_rate

    effective_warmup = min(max(warmup_steps, 0), total_steps)
    if effective_warmup > 0 and current_step <= effective_warmup:
        return base_learning_rate * current_step / effective_warmup

    if total_steps <= effective_warmup + 1:
        return base_learning_rate

    min_lr = base_learning_rate * min_lr_ratio
    decay_steps = total_steps - effective_warmup - 1
    progress = (current_step - effective_warmup - 1) / max(decay_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_learning_rate - min_lr) * cosine


def set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def format_train_log(
    step: int,
    loss: float,
    learning_rate: float,
) -> str:
    """
    Format one train-step log line.

    Output:
        log_line: str
    """

    return f"step={step} loss={loss:.6f} lr={learning_rate:.8f}"


def append_train_metric(
    metrics_path: str,
    step: int,
    loss: float,
    learning_rate: float,
) -> None:
    """
    Append one train metric record into a local jsonl file.

    Input:
        metrics_path: str
        step: int
        loss: float
        learning_rate: float

    Output:
        none

    Expected side effect:
        One json line is appended to metrics_path, containing at least:
        - "step"
        - "loss"
        - "learning_rate"

    Recommended path convention:
        metrics_path should usually live under logs/, for example:
        logs/pretrain_metrics.jsonl
    """

    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "step": step,
        "loss": loss,
        "learning_rate": learning_rate,
    }
    with metrics_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_training_state(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> int:
    """
    Restore model / optimizer / step from one checkpoint file.

    Input:
        checkpoint_path: str
        model: torch.nn.Module
        optimizer: torch.optim.Optimizer
        device: str

    Output:
        start_step: int
    """

    checkpoint = load_checkpoint_file(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return int(checkpoint["step"])


def _build_formal_checkpoint_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    batch_in_epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, Any]:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
    }
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint


def _save_resume_checkpoint(
    checkpoint_dir: str,
    save_weight: str,
    checkpoint: Dict[str, Any],
) -> Path:
    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_root / f"{save_weight}_resume_latest.pt"
    tmp_path = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, checkpoint_path)
    return checkpoint_path


def _normalize_resume_position(
    epoch: int,
    batch_in_epoch: int,
    total_batches: Optional[int],
) -> Tuple[int, int]:
    if total_batches is not None and batch_in_epoch >= total_batches:
        return epoch + 1, 0
    return epoch, batch_in_epoch


def _build_epoch_dataloader(
    dataset: Any,
    fallback_dataloader: Iterable[Dict[str, torch.Tensor]],
    batch_size: int,
    num_workers: int,
    epoch: int,
) -> Tuple[Iterable[Dict[str, torch.Tensor]], Optional[int]]:
    if hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__"):
        generator = torch.Generator()
        generator.manual_seed(42 + epoch)
        dataloader = build_pretrain_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
        )
        return dataloader, len(dataloader)

    total_batches = len(fallback_dataloader) if hasattr(fallback_dataloader, "__len__") else None
    return fallback_dataloader, total_batches


def _estimate_batches_per_epoch(
    dataset: Any,
    fallback_dataloader: Iterable[Dict[str, torch.Tensor]],
    batch_size: int,
) -> Optional[int]:
    if hasattr(dataset, "__len__"):
        return math.ceil(len(dataset) / batch_size)
    if hasattr(fallback_dataloader, "__len__"):
        return len(fallback_dataloader)
    return None


def _estimate_total_update_steps(
    batches_per_epoch: Optional[int],
    epochs: int,
    accumulation_steps: int,
) -> int:
    if batches_per_epoch is None:
        return 0
    return math.ceil(batches_per_epoch / accumulation_steps) * epochs


def run_formal_pretrain(args: argparse.Namespace) -> List[float]:
    """
    Run the formal pretrain entry from parsed args.

    Input:
        args: argparse.Namespace

    Output:
        loss_history: list[float]

    Recommended output convention:
        - logs/ for metric jsonl files
        - checkpoints/ for resume checkpoints
        - out/ for final exported weights
    """
    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive.")
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0.0 and 1.0.")
    if args.from_resume != "none" and args.from_weight != "none":
        raise ValueError("from_resume and from_weight cannot both be set.")

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    runtime = build_runtime_from_args(args)
    model = runtime["model"]
    dataset = runtime["dataset"]
    dataloader = runtime["dataloader"]
    optimizer = runtime["optimizer"]
    scaler = build_grad_scaler(args.device, args.dtype)

    batches_per_epoch = _estimate_batches_per_epoch(
        dataset=dataset,
        fallback_dataloader=dataloader,
        batch_size=args.batch_size,
    )
    total_update_steps = _estimate_total_update_steps(
        batches_per_epoch=batches_per_epoch,
        epochs=args.epochs,
        accumulation_steps=args.accumulation_steps,
    )

    metrics_path = Path(args.log_dir) / f"{args.save_weight}_metrics.jsonl"
    text_log_path = Path(args.log_dir) / f"{args.save_weight}.log"

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

    loss_history: List[float] = []
    accum_loss = 0.0
    accum_count = 0
    current_lr = optimizer.param_groups[0]["lr"]

    optimizer.zero_grad()

    for epoch in range(resume_epoch, args.epochs):
        epoch_dataloader, total_batches = _build_epoch_dataloader(
            dataset=dataset,
            fallback_dataloader=dataloader,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epoch=epoch,
        )
        start_batch_idx = resume_batch_idx if epoch == resume_epoch else 0

        for batch_idx, batch in enumerate(epoch_dataloader):
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
                loss = outputs.loss

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
                    total_batches=total_batches,
                )
                _save_resume_checkpoint(
                    checkpoint_dir=args.checkpoint_dir,
                    save_weight=args.save_weight,
                    checkpoint=_build_formal_checkpoint_state(
                        model=model,
                        optimizer=optimizer,
                        step=update_step,
                        epoch=next_epoch,
                        batch_in_epoch=next_batch_in_epoch,
                        scaler=scaler,
                    ),
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

            if args.save_interval > 0 and update_step % args.save_interval == 0:
                _save_resume_checkpoint(
                    checkpoint_dir=args.checkpoint_dir,
                    save_weight=args.save_weight,
                    checkpoint=_build_formal_checkpoint_state(
                        model=model,
                        optimizer=optimizer,
                        step=update_step,
                        epoch=epoch + 1,
                        batch_in_epoch=0,
                        scaler=scaler,
                    ),
                )

            accum_loss = 0.0
            accum_count = 0

        resume_batch_idx = 0

    _save_resume_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        save_weight=args.save_weight,
        checkpoint=_build_formal_checkpoint_state(
            model=model,
            optimizer=optimizer,
            step=update_step,
            epoch=args.epochs,
            batch_in_epoch=0,
            scaler=scaler,
        ),
    )

    final_weight_path = Path(args.out_dir) / f"{args.save_weight}_final.pt"
    torch.save(model.state_dict(), final_weight_path)

    return loss_history

def main(argv: Optional[List[str]] = None) -> None:
    """
    Main entry for the formal pretrain launcher.
    """
    args = parse_train_args(argv)
    run_formal_pretrain(args)


if __name__ == "__main__":
    main()
