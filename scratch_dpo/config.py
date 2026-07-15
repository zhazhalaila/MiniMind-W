from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DPODataConfig:
    """
    DPO data configuration container.

    Fields:
        tokenizer_dir: local tokenizer directory.
        data_path: local DPO jsonl path.
        max_seq_len: max sequence length before x/y shift.
        empty_think_ratio: probability of keeping empty thinking block.
    """

    tokenizer_dir: str
    data_path: str
    max_seq_len: int
    empty_think_ratio: float = 0.2


@dataclass(frozen=True)
class DPOTrainConfig:
    """
    DPO train configuration container.

    Fields:
        log_dir: text / jsonl metric output directory.
        checkpoint_dir: latest resume checkpoint directory.
        out_dir: exported model weight directory.
        save_weight: exported weight prefix.
        from_weight: SFT weight path used to initialize policy and reference model.
        from_resume: latest training-state checkpoint path.
        epochs: number of passes over dataset.
        batch_size: number of preference pairs per batch.
        learning_rate: optimizer learning rate.
        weight_decay: AdamW weight decay.
        device: target device string.
        dtype: autocast dtype name.
        num_workers: dataloader workers.
        accumulation_steps: gradient accumulation steps.
        grad_clip: gradient norm clipping threshold.
        log_interval: metric logging interval in update steps.
        save_interval: latest checkpoint overwrite interval in update steps.
        warmup_steps: scheduler warmup update steps.
        min_lr_ratio: final lr ratio for cosine schedule.
        beta: DPO beta coefficient.
    """

    log_dir: str
    checkpoint_dir: str
    out_dir: str
    save_weight: str
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
    beta: float


def build_dpo_data_config(
    tokenizer_dir: str,
    data_path: str,
    max_seq_len: int,
    empty_think_ratio: float = 0.2,
) -> DPODataConfig:
    """
    Build one DPO data config object.

    Input:
        tokenizer_dir: str
        data_path: str
        max_seq_len: int
        empty_think_ratio: float

    Output:
        DPODataConfig
    """

    return DPODataConfig(
        tokenizer_dir=tokenizer_dir,
        data_path=data_path,
        max_seq_len=max_seq_len,
        empty_think_ratio=empty_think_ratio,
    )


def build_dpo_train_config(
    log_dir: str,
    checkpoint_dir: str,
    out_dir: str,
    save_weight: str,
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
    beta: float,
) -> DPOTrainConfig:
    """
    Build one DPO train config object.

    Output:
        DPOTrainConfig
    """

    return DPOTrainConfig(
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        out_dir=out_dir,
        save_weight=save_weight,
        from_weight=from_weight,
        from_resume=from_resume,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        dtype=dtype,
        num_workers=num_workers,
        accumulation_steps=accumulation_steps,
        grad_clip=grad_clip,
        log_interval=log_interval,
        save_interval=save_interval,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
        beta=beta,
    )