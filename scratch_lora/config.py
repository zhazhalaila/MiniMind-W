from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LoRADataConfig:
    """
    LoRA SFT data configuration.

    Fields:
        tokenizer_dir: local tokenizer directory.
        data_path: local SFT-style jsonl path for LoRA fine-tuning.
        max_seq_len: max sequence length for one full conversation sample.
        add_system_ratio: probability of adding system prompt.
        empty_think_ratio: probability of keeping empty thinking block.
    """

    tokenizer_dir: str
    data_path: str
    max_seq_len: int
    add_system_ratio: float = 0.2
    empty_think_ratio: float = 0.2


@dataclass(frozen=True)
class LoRATrainConfig:
    """
    LoRA train configuration.

    Fields:
        log_dir: text / jsonl metric output directory.
        checkpoint_dir: latest resume checkpoint directory.
        out_dir: exported LoRA weight directory.
        lora_name: LoRA weight prefix.
        from_weight: base model weight path.
        from_resume: latest training-state checkpoint path.
        epochs: number of passes over dataset.
        batch_size: number of SFT samples per batch.
        learning_rate: optimizer learning rate.
        weight_decay: AdamW weight decay.
        device: target device string.
        dtype: autocast dtype name.
        num_workers: dataloader workers.
        accumulation_steps: gradient accumulation steps.
        grad_clip: gradient norm clipping threshold.
        log_interval: metric logging interval.
        save_interval: latest checkpoint overwrite interval.
        warmup_steps: scheduler warmup update steps.
        min_lr_ratio: final lr ratio for cosine schedule.
        rank: LoRA low-rank dimension.
        target_modules: optional comma-separated target module name fragments.
    """

    log_dir: str
    checkpoint_dir: str
    out_dir: str
    lora_name: str
    from_weight: str
    from_resume: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    device: str
    dtype: str
    num_workers: int
    accumulation_steps: int
    grad_clip: float
    log_interval: int
    save_interval: int
    warmup_steps: int
    min_lr_ratio: float
    rank: int = 16
    target_modules: Optional[str] = None


def build_lora_data_config(
    tokenizer_dir: str,
    data_path: str,
    max_seq_len: int,
    add_system_ratio: float = 0.2,
    empty_think_ratio: float = 0.2,
) -> LoRADataConfig:
    """
    Build LoRA data config.

    Input:
        tokenizer_dir: str
        data_path: str
        max_seq_len: int
        add_system_ratio: float
        empty_think_ratio: float

    Output:
        LoRADataConfig
    """

    if not isinstance(tokenizer_dir, str) or not tokenizer_dir:
        raise ValueError("tokenizer_dir must be a non-empty string.")

    if not isinstance(data_path, str) or not data_path:
        raise ValueError("data_path must be a non-empty string.")

    if not isinstance(max_seq_len, int) or max_seq_len <= 0:
        raise ValueError("max_seq_len must be a positive integer.")

    if not isinstance(add_system_ratio, (int, float)) or not 0.0 <= float(add_system_ratio) <= 1.0:
        raise ValueError("add_system_ratio must be in [0, 1].")

    if not isinstance(empty_think_ratio, (int, float)) or not 0.0 <= float(empty_think_ratio) <= 1.0:
        raise ValueError("empty_think_ratio must be in [0, 1].")

    return LoRADataConfig(
        tokenizer_dir=tokenizer_dir,
        data_path=data_path,
        max_seq_len=max_seq_len,
        add_system_ratio=float(add_system_ratio),
        empty_think_ratio=float(empty_think_ratio),
    )


def build_lora_train_config(
    log_dir: str,
    checkpoint_dir: str,
    out_dir: str,
    lora_name: str,
    from_weight: str,
    from_resume: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
    dtype: str,
    num_workers: int,
    accumulation_steps: int,
    grad_clip: float,
    log_interval: int,
    save_interval: int,
    warmup_steps: int,
    min_lr_ratio: float,
    rank: int = 16,
    target_modules: Optional[str] = None,
) -> LoRATrainConfig:
    """
    Build LoRA train config.

    Output:
        LoRATrainConfig
    """

    if not isinstance(log_dir, str) or not log_dir:
        raise ValueError("log_dir must be a non-empty string.")

    if not isinstance(checkpoint_dir, str) or not checkpoint_dir:
        raise ValueError("checkpoint_dir must be a non-empty string.")

    if not isinstance(out_dir, str) or not out_dir:
        raise ValueError("out_dir must be a non-empty string.")

    if not isinstance(lora_name, str) or not lora_name:
        raise ValueError("lora_name must be a non-empty string.")

    if not isinstance(from_weight, str) or not from_weight:
        raise ValueError("from_weight must be a non-empty string.")

    if not isinstance(from_resume, str) or not from_resume:
        raise ValueError("from_resume must be a non-empty string.")

    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer.")

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    if not isinstance(learning_rate, (int, float)) or float(learning_rate) <= 0:
        raise ValueError("learning_rate must be positive.")

    if not isinstance(weight_decay, (int, float)) or float(weight_decay) < 0:
        raise ValueError("weight_decay must be non-negative.")

    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty string.")

    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("dtype must be one of: float32, float16, bfloat16.")

    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer.")

    if not isinstance(accumulation_steps, int) or accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be a positive integer.")

    if not isinstance(grad_clip, (int, float)) or float(grad_clip) < 0:
        raise ValueError("grad_clip must be non-negative.")

    if not isinstance(log_interval, int) or log_interval <= 0:
        raise ValueError("log_interval must be a positive integer.")

    if not isinstance(save_interval, int) or save_interval <= 0:
        raise ValueError("save_interval must be a positive integer.")

    if not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError("warmup_steps must be a non-negative integer.")

    if not isinstance(min_lr_ratio, (int, float)) or not 0.0 <= float(min_lr_ratio) <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1].")

    if not isinstance(rank, int) or rank <= 0:
        raise ValueError("rank must be a positive integer.")

    if target_modules is not None and not isinstance(target_modules, str):
        raise ValueError("target_modules must be a string or None.")

    return LoRATrainConfig(
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        out_dir=out_dir,
        lora_name=lora_name,
        from_weight=from_weight,
        from_resume=from_resume,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        device=device,
        dtype=dtype,
        num_workers=num_workers,
        accumulation_steps=accumulation_steps,
        grad_clip=float(grad_clip),
        log_interval=log_interval,
        save_interval=save_interval,
        warmup_steps=warmup_steps,
        min_lr_ratio=float(min_lr_ratio),
        rank=rank,
        target_modules=target_modules,
    )

