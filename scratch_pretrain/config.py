from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PretrainDataConfig:
    """
    Pretrain data configuration container.

    Fields:
        tokenizer_dir: str
            Input: local tokenizer directory path.
            Output: used by tokenizer loading code.

        data_path: str
            Input: local jsonl path for pretrain text data.
            Output: used by dataset loading code.

        max_length: int
            Input: target fixed sequence length.
            Output: used by encoding / padding logic.
    """

    tokenizer_dir: str
    data_path: str
    max_length: int


def build_pretrain_data_config(tokenizer_dir: str, data_path: str, max_length: int) -> PretrainDataConfig:
    """
    Build the pretrain data config object.

    Input:
        tokenizer_dir: str
        data_path: str
        max_length: int

    Output:
        PretrainDataConfig
    """

    return PretrainDataConfig(
        tokenizer_dir=tokenizer_dir,
        data_path=data_path,
        max_length=max_length
    )


@dataclass(frozen=True)
class PretrainTrainConfig:
    """
    Pretrain train-loop configuration container.

    Fields:
        save_dir: str
            Input: local directory path for checkpoints.
            Output: used by checkpoint saving code.

        batch_size: int
            Input shape meaning: scalar.
            Output role: dataloader batch size B.

        learning_rate: float
            Input shape meaning: scalar.
            Output role: optimizer learning rate.

        weight_decay: float
            Input shape meaning: scalar.
            Output role: optimizer weight decay.

        max_steps: int
            Input shape meaning: scalar.
            Output role: maximum number of train steps.

        device: str
            Input: device string such as "cpu" or "cuda:0".
            Output: used by batch / model device placement code.

        log_every: int
            Input shape meaning: scalar.
            Output role: logging frequency in steps.

        save_every: int
            Input shape meaning: scalar.
            Output role: checkpoint frequency in steps.
    """

    save_dir: str
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_steps: int
    device: str
    log_every: int
    save_every: int


def build_pretrain_train_config(
    save_dir: str,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_steps: int,
    device: str,
    log_every: int,
    save_every: int,
) -> PretrainTrainConfig:
    """
    Build the pretrain train-loop config object.

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
        PretrainTrainConfig
    """

    return PretrainTrainConfig(
        save_dir=save_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_steps=max_steps,
        device=device,
        log_every=log_every,
        save_every=save_every,
    )