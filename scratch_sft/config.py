from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SFTDataConfig:
    """
    SFT data configuration container.

    Fields:
        tokenizer_dir: str
            Input: local tokenizer directory path.
            Output: used by tokenizer loading code.

        data_path: str
            Input: local jsonl path for SFT data.
            Output: used by dataset loading code.

        max_seq_len: int
            Input: target fixed sequence length.
            Output: used by encoding / truncation / padding logic.

        add_system_ratio: float
            Input: probability of prepending one synthetic system turn
            when the record does not already start with `system`.
            Output: used by MiniMind-style conversation preprocessing.

        empty_think_ratio: float
            Input: probability of keeping an empty `<think>` block.
            Output: used by MiniMind-style prompt postprocessing.
    """

    tokenizer_dir: str
    data_path: str
    max_seq_len: int
    add_system_ratio: float = 0.2
    empty_think_ratio: float = 0.2


@dataclass(frozen=True)
class SFTTrainConfig:
    """
    SFT train configuration container.

    Fields:
        save_dir: str
        batch_size: int
        learning_rate: float
        weight_decay: float
        max_steps: int
        device: str
        log_every: int
        save_every: int
    """

    save_dir: str
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_steps: int
    device: str
    log_every: int
    save_every: int


def build_sft_data_config(
    tokenizer_dir: str,
    data_path: str,
    max_seq_len: int,
    add_system_ratio: float = 0.2,
    empty_think_ratio: float = 0.2,
) -> SFTDataConfig:
    """
    Build one SFT data config object.

    Input:
        tokenizer_dir: str
        data_path: str
        max_seq_len: int
        add_system_ratio: float
        empty_think_ratio: float

    Output:
        SFTDataConfig
    """

    if not isinstance(tokenizer_dir, str) or not tokenizer_dir.strip():
        raise ValueError("tokenizer_dir must be a non-empty string.")
    
    if not isinstance(data_path, str) or not data_path.strip():
        raise ValueError("data_path must be a non-empty string.")

    if not isinstance(max_seq_len, int) or max_seq_len <= 0:
        raise ValueError("max_seq_len must be a positive integer.")

    if not isinstance(add_system_ratio, (int, float)) or not 0.0 <= float(add_system_ratio) <= 1.0:
        raise ValueError("add_system_ratio must be in [0, 1].")

    if not isinstance(empty_think_ratio, (int, float)) or not 0.0 <= float(empty_think_ratio) <= 1.0:
        raise ValueError("empty_think_ratio must be in [0, 1].")

    return SFTDataConfig(
        tokenizer_dir=tokenizer_dir,
        data_path=data_path,
        max_seq_len=max_seq_len,
        add_system_ratio=float(add_system_ratio),
        empty_think_ratio=float(empty_think_ratio),
    )

def build_sft_train_config(
    save_dir: str,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_steps: int,
    device: str,
    log_every: int,
    save_every: int,
) -> SFTTrainConfig:
    """
    Build one SFT train config object.

    Input:
        save_dir: str
        batch_size: int
        learning_rate: float
        weight_decay: float
        max_steps: int
        device: str
        log_every: int
        save_every: int

    Output:
        SFTTrainConfig
    """

    if not isinstance(save_dir, str) or not save_dir.strip():
        raise ValueError("save_dir must be a non-empty string.")

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")

    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer.")

    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty string.")

    if not isinstance(log_every, int) or log_every <= 0:
        raise ValueError("log_every must be a positive integer.")

    if not isinstance(save_every, int) or save_every <= 0:
        raise ValueError("save_every must be a positive integer.")

    return SFTTrainConfig(
        save_dir=save_dir,
        batch_size=batch_size,
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        max_steps=max_steps,
        device=device,
        log_every=log_every,
        save_every=save_every,
    )
