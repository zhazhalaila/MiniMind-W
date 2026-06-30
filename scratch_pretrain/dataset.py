from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset


def load_jsonl_records(data_path: str) -> List[Dict[str, Any]]:
    """
    Load a jsonl file into a list of python dict records.

    Input:
        data_path: str

    Output:
        records: list[dict[str, Any]]
    """

    records = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records


def build_pretrain_example(sample: Dict[str, Any], tokenizer: Any, max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert one raw pretrain sample into model-ready tensors.

    Input:
        sample: dict[str, Any]
            Expected to contain at least:
            - sample["text"]: str
        tokenizer: Any
        max_length: int

    Output:
        input_ids: torch.Tensor
            Shape: (max_length,)
            Dtype: torch.long

        labels: torch.Tensor
            Shape: (max_length,)
            Dtype: torch.long

    Expected internal steps:
        1. Read sample["text"].
        2. Tokenize with:
           tokenizer(text, add_special_tokens=False, max_length=max_length - 2, truncation=True)
        3. Add BOS and EOS.
        4. Pad to max_length with pad_token_id.
        5. Copy input_ids into labels.
        6. Replace padding positions in labels with -100.
    """

    text = str(sample["text"])

    tokens = tokenizer(
        text,
        add_special_tokens=False,
        max_length=max_length - 2,
        truncation=True,
    ).input_ids

    tokens = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]

    input_ids = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))
    input_ids = torch.tensor(input_ids, dtype=torch.long)

    labels = input_ids.clone()
    labels[input_ids == tokenizer.pad_token_id] = -100

    return input_ids, labels


class PretrainDataset(Dataset):
    """
    Minimal pretrain dataset skeleton.

    Input:
        data_path: str
        tokenizer: Any
        max_length: int

    Output from __getitem__:
        tuple[torch.Tensor, torch.Tensor]
            input_ids, labels

    MiniMind alignment:
        - tokenizer is passed in from outside
        - __getitem__ should directly produce (input_ids, labels)
    """

    def __init__(self, data_path: str, tokenizer: Any, max_length: int = 512):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = load_jsonl_records(str(self.data_path))

    def __len__(self) -> int:
        """
        Input:
            none

        Output:
            dataset_size: int
        """

        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            index: int

        Output:
            input_ids: torch.Tensor
            labels: torch.Tensor

        Suggested implementation:
            sample = self.records[index]
            return build_pretrain_example(sample, self.tokenizer, self.max_length)
        """

        sample = self.records[index]
        return build_pretrain_example(sample, self.tokenizer, self.max_length)
