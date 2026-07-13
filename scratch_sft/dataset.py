from __future__ import annotations

from typing import Any, Dict, List

import torch, json
from torch.utils.data import Dataset

from scratch_sft.prompt import (
    format_sft_messages,
    build_sft_chat_prompt,
    build_sft_special_token_ids,
    generate_sft_labels,
    pad_sft_example,
    postprocess_sft_prompt,
)


def load_sft_jsonl_records(data_path: str) -> List[Dict[str, Any]]:
    """
    Load SFT jsonl records from local disk.

    Input:
        data_path: str

    Output:
        records: list[dict[str, Any]]
    """

    if not isinstance(data_path, str) or not data_path.strip():
        raise ValueError("data_path must be a non-empty string.")
    
    records: list[dict[str, Any]] = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc

            if not isinstance(record, dict):
                raise ValueError(f"Line {line_no} must be a JSON object.")

            if "conversations" not in record:
                raise ValueError(f"Line {line_no} must contain 'conversations'.")

            if not isinstance(record["conversations"], list) or not record["conversations"]:
                raise ValueError(f"Line {line_no} conversations must be a non-empty list.")

            records.append(record)

    return records


def build_sft_example(
    record: Dict[str, Any],
    tokenizer: Any,
    assistant_bos_ids: list[int],
    assistant_eos_ids: list[int],
    max_seq_len: int,
    add_system_ratio: float = 0.2,
    empty_think_ratio: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """
    Build one tensorized SFT example from one raw record.

    Input:
        record: dict[str, Any]
            Expected:
                record["conversations"]: list[dict[str, Any]]
        tokenizer: Any
        assistant_bos_ids: list[int]
        assistant_eos_ids: list[int]
        max_seq_len: int
        add_system_ratio: float
        empty_think_ratio: float

    Output:
        example: dict[str, torch.Tensor]
            example["input_ids"].shape == (L,)
            example["labels"].shape == (L,)
    """

    if not isinstance(max_seq_len, int) or max_seq_len <= 0:
        raise ValueError("max_seq_len must be a positive integer.")
    
    # add system prompt by random
    messages = format_sft_messages(
        record=record,
        add_system_ratio=add_system_ratio,
    )

    # normalize prompt
    prompt_text = build_sft_chat_prompt(
        messages=messages,
        tokenizer=tokenizer,
    )

    # delete random empty tag
    prompt_text = postprocess_sft_prompt(
        prompt_text=prompt_text,
        empty_think_ratio=empty_think_ratio,
    )

    # transfer prompt (str) into token (int)
    input_ids = tokenizer(prompt_text).input_ids[:max_seq_len]

    # mask promp for labels, loss value works on assistant response
    labels = generate_sft_labels(
        input_ids=input_ids,
        assistant_bos_ids=assistant_bos_ids,
        assistant_eos_ids=assistant_eos_ids,
        max_seq_len=max_seq_len,
    )

    # padding 
    input_ids, labels = pad_sft_example(
        input_ids=input_ids,
        labels=labels,
        pad_token_id=tokenizer.pad_token_id,
        max_seq_len=max_seq_len,
    )

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class SFTDataset(Dataset):
    """
    Minimal SFT dataset.

    Single-sample output:
        dict[str, torch.Tensor]
            input_ids.shape == (L,)
            labels.shape == (L,)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_seq_len: int,
        add_system_ratio: float = 0.2,
        empty_think_ratio: float = 0.2,
    ) -> None:
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.add_system_ratio = add_system_ratio
        self.empty_think_ratio = empty_think_ratio
        
        self.records = load_sft_jsonl_records(data_path)

        self.assistant_bos_ids, self.assistant_eos_ids = build_sft_special_token_ids(
            tokenizer
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not isinstance(idx, int):
            raise TypeError("idx must be an int.")
        
        record = self.records[idx]

        # return Tensor
        return build_sft_example(
            record=record,
            tokenizer=self.tokenizer,
            assistant_bos_ids=self.assistant_bos_ids,
            assistant_eos_ids=self.assistant_eos_ids,
            max_seq_len=self.max_seq_len,
            add_system_ratio=self.add_system_ratio,
            empty_think_ratio=self.empty_think_ratio,
        )
