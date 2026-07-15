from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch, json, random
from torch.utils.data import Dataset


def _extract_input_ids(tokenizer_output: Any) -> List[int]:
    if isinstance(tokenizer_output, dict):
        return tokenizer_output["input_ids"]
    return tokenizer_output.input_ids


def load_dpo_jsonl_records(data_path: str) -> List[Dict[str, Any]]:
    """
    Load local MiniMind-style DPO jsonl records.

    Input:
        data_path: str

    Output:
        records: list[dict[str, Any]]
            Each record is expected to contain:
            - record["chosen"]: list[dict[str, Any]]
            - record["rejected"]: list[dict[str, Any]]
    """

    records: list[dict[str, Any]] = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            record = json.loads(line)

            if "chosen" not in record or "rejected" not in record:
                raise ValueError("Each DPO record must contain 'chosen' and 'rejected'.")
            
            records.append(record)

    return records


def build_dpo_special_token_ids(tokenizer: Any) -> Tuple[List[int], List[int]]:
    """
    Build assistant span marker token ids, aligned with MiniMind DPODataset.

    Input:
        tokenizer

    Output:
        assistant_bos_ids: list[int]
        assistant_eos_ids: list[int]
    """

    assistant_bos_ids = _extract_input_ids(
        tokenizer(
            f"{tokenizer.bos_token}assistant\n",
            add_special_tokens=False,
        )
    )

    assistant_eos_ids = _extract_input_ids(
        tokenizer(
            f"{tokenizer.eos_token}\n",
            add_special_tokens=False,
        )
    )

    return assistant_bos_ids, assistant_eos_ids


def build_dpo_chat_prompt(messages: List[Dict[str, Any]], tokenizer: Any) -> str:
    """
    Convert chosen / rejected messages into one chat-template string.

    Input:
        messages: list[dict[str, Any]]
        tokenizer

    Output:
        prompt_text: str
    """

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def postprocess_dpo_prompt(prompt_text: str, empty_think_ratio: float = 0.2) -> str:
    """
    Postprocess chat-template text before tokenization.

    Input:
        prompt_text: str
        empty_think_ratio: float

    Output:
        processed_prompt_text: str
    """

    if empty_think_ratio <= 0:
        return prompt_text
    
    if random.random() >= empty_think_ratio:
        return prompt_text
    
    return prompt_text.replace("<think>\n\n</think>\n\n", "")


def generate_dpo_loss_mask(
    input_ids: List[int],
    assistant_bos_ids: List[int],
    assistant_eos_ids: List[int],
    max_seq_len: int,
) -> List[int]:
    """
    Build token-level loss mask for assistant spans.

    Input:
        input_ids: list[int]
            Shape: (L,)
        assistant_bos_ids: list[int]
        assistant_eos_ids: list[int]
        max_seq_len: int

    Output:
        loss_mask: list[int]
            Shape: (L,)
            1 means this target token participates in DPO log-prob sum.
    """
    loss_mask = [0] * len(input_ids)

    i = 0
    while i < len(input_ids):
        if input_ids[i : i + len(assistant_bos_ids)] == assistant_bos_ids:
            start = i + len(assistant_bos_ids)
            end = start

            while end < len(input_ids):
                if input_ids[end : end + len(assistant_eos_ids)] == assistant_eos_ids:
                    break
                end += 1

            label_end = min(
                end + len(assistant_eos_ids),
                max_seq_len,
                len(input_ids),
            )

            for j in range(start, label_end):
                loss_mask[j] = 1

            if end < len(input_ids):
                i = end + len(assistant_eos_ids)
            else:
                i = len(input_ids)
        else:
            i += 1

    return loss_mask


def build_dpo_sequence_tensors(
    input_ids: List[int],
    loss_mask: List[int],
    pad_token_id: int,
    max_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert one full token sequence into shifted x/y/mask tensors.

    Input:
        input_ids: list[int]
            Shape: (L,)
        loss_mask: list[int]
            Shape: (L,)
        pad_token_id: int
        max_seq_len: int

    Output:
        x: torch.Tensor
            Shape: (max_seq_len - 1,)
        y: torch.Tensor
            Shape: (max_seq_len - 1,)
        mask: torch.Tensor
            Shape: (max_seq_len - 1,)
    """

    input_ids = input_ids[:max_seq_len]
    loss_mask = loss_mask[:max_seq_len]

    pad_len = max_seq_len - len(input_ids)

    if pad_len > 0:
        input_ids = input_ids + [pad_token_id] * pad_len
        loss_mask = loss_mask + [0] * pad_len

    # start fom bos, end before eos
    x = torch.tensor(input_ids[:-1], dtype=torch.long)
    # start from the first token, end at eos
    y = torch.tensor(input_ids[1:], dtype=torch.long)
    mask = torch.tensor(loss_mask[1:], dtype=torch.float32)

    return x, y, mask


def build_dpo_pair_example(
    record: Dict[str, Any],
    tokenizer: Any,
    assistant_bos_ids: List[int],
    assistant_eos_ids: List[int],
    max_seq_len: int,
    empty_think_ratio: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """
    Convert one raw DPO record into MiniMind-style chosen / rejected tensors.

    Input:
        record: dict[str, Any]
            Expected keys:
            - "chosen": list[dict[str, Any]]
            - "rejected": list[dict[str, Any]]
        tokenizer
        assistant_bos_ids: list[int]
        assistant_eos_ids: list[int]
        max_seq_len: int
        empty_think_ratio: float

    Output:
        example: dict[str, torch.Tensor]
            x_chosen.shape == (L,)
            y_chosen.shape == (L,)
            mask_chosen.shape == (L,)
            x_rejected.shape == (L,)
            y_rejected.shape == (L,)
            mask_rejected.shape == (L,)
            where L = max_seq_len - 1
    """

    chosen_prompt = build_dpo_chat_prompt(record["chosen"], tokenizer)
    rejected_prompt = build_dpo_chat_prompt(record["rejected"], tokenizer)

    chosen_prompt = postprocess_dpo_prompt(
        chosen_prompt,
        empty_think_ratio=empty_think_ratio,
    )
    rejected_prompt = postprocess_dpo_prompt(
        rejected_prompt,
        empty_think_ratio=empty_think_ratio,
    )

    chosen_input_ids = _extract_input_ids(
        tokenizer(
            chosen_prompt,
            add_special_tokens=False,
        )
    )

    rejected_input_ids = _extract_input_ids(
        tokenizer(
            rejected_prompt,
            add_special_tokens=False,
        )
    )

    chosen_loss_mask = generate_dpo_loss_mask(
        input_ids=chosen_input_ids,
        assistant_bos_ids=assistant_bos_ids,
        assistant_eos_ids=assistant_eos_ids,
        max_seq_len=max_seq_len,
    )

    rejected_loss_mask = generate_dpo_loss_mask(
        input_ids=rejected_input_ids,
        assistant_bos_ids=assistant_bos_ids,
        assistant_eos_ids=assistant_eos_ids,
        max_seq_len=max_seq_len,
    )

    x_chosen, y_chosen, mask_chosen = build_dpo_sequence_tensors(
        input_ids=chosen_input_ids,
        loss_mask=chosen_loss_mask,
        pad_token_id=tokenizer.pad_token_id,
        max_seq_len=max_seq_len,
    )

    x_rejected, y_rejected, mask_rejected = build_dpo_sequence_tensors(
        input_ids=rejected_input_ids,
        loss_mask=rejected_loss_mask,
        pad_token_id=tokenizer.pad_token_id,
        max_seq_len=max_seq_len,
    )

    return {
        "x_chosen": x_chosen,
        "y_chosen": y_chosen,
        "mask_chosen": mask_chosen,
        "x_rejected": x_rejected,
        "y_rejected": y_rejected,
        "mask_rejected": mask_rejected,
    }


class DPODataset(Dataset):
    """
    MiniMind-style DPO dataset.

    Single-sample output:
        dict[str, torch.Tensor]
            x_chosen.shape == (L,)
            y_chosen.shape == (L,)
            mask_chosen.shape == (L,)
            x_rejected.shape == (L,)
            y_rejected.shape == (L,)
            mask_rejected.shape == (L,)
    """

    def __init__(
        self,
        records: Union[str, Path, List[Dict[str, Any]]],
        tokenizer,
        max_seq_len: int,
        empty_think_ratio: float = 0.2,
    ) -> None:
        if isinstance(records, (str, Path)):
            self.records = load_dpo_jsonl_records(str(records))
        else:
            self.records = records
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.empty_think_ratio = empty_think_ratio

        self.assistant_bos_ids, self.assistant_eos_ids = build_dpo_special_token_ids(
            tokenizer
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]

        return build_dpo_pair_example(
            record=record,
            tokenizer=self.tokenizer,
            assistant_bos_ids=self.assistant_bos_ids,
            assistant_eos_ids=self.assistant_eos_ids,
            max_seq_len=self.max_seq_len,
            empty_think_ratio=self.empty_think_ratio,
        )
