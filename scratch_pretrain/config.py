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