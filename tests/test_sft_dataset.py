import json

import pytest
import torch

from scratch_sft.dataset import SFTDataset, build_sft_example, load_sft_jsonl_records
from scratch_sft.prompt import build_sft_special_token_ids


def xfail_on_not_implemented(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.xfail(f"{fn.__name__} is not implemented yet.")


class FakeTokenizer:
    bos_token = "<bos>"
    eos_token = "<eos>"
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None):
        del tokenize
        del tools
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        prompt = "\n".join(parts)
        if add_generation_prompt:
            prompt += "\nassistant: "
        return prompt

    def __call__(self, text, add_special_tokens=False, max_length=None, truncation=False):
        del add_special_tokens
        token_ids = [ord(ch) % 50 + 10 for ch in text]
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        return type("Tokenized", (), {"input_ids": token_ids})()


def test_load_sft_jsonl_records(tmp_path):
    path = tmp_path / "toy_sft.jsonl"
    rows = [
        {"conversations": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
        {"conversations": [{"role": "user", "content": "where"}, {"role": "assistant", "content": "Zhejiang"}]},
    ]
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    actual = xfail_on_not_implemented(load_sft_jsonl_records, str(path))
    assert isinstance(actual, list)
    assert len(actual) == 2


def test_build_sft_example_returns_rank1_tensors():
    tokenizer = FakeTokenizer()
    record = {
        "conversations": [
            {"role": "user", "content": "Introduce Hangzhou."},
            {"role": "assistant", "content": "Hangzhou is the capital of Zhejiang."},
        ],
    }
    assistant_bos_ids, assistant_eos_ids = xfail_on_not_implemented(build_sft_special_token_ids, tokenizer)
    actual = xfail_on_not_implemented(
        build_sft_example,
        record=record,
        tokenizer=tokenizer,
        assistant_bos_ids=assistant_bos_ids,
        assistant_eos_ids=assistant_eos_ids,
        max_seq_len=24,
        add_system_ratio=0.0,
        empty_think_ratio=0.2,
    )
    assert isinstance(actual, dict)
    assert set(actual.keys()) == {"input_ids", "labels"}
    assert isinstance(actual["input_ids"], torch.Tensor)
    assert isinstance(actual["labels"], torch.Tensor)
    assert actual["input_ids"].ndim == 1
    assert actual["labels"].ndim == 1


def test_sft_dataset_returns_single_example_dict(tmp_path):
    path = tmp_path / "toy_sft.jsonl"
    rows = [
        {"conversations": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
        {"conversations": [{"role": "user", "content": "where"}, {"role": "assistant", "content": "Zhejiang"}]},
    ]
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    tokenizer = FakeTokenizer()
    dataset = xfail_on_not_implemented(
        SFTDataset,
        data_path=str(path),
        tokenizer=tokenizer,
        max_seq_len=24,
        add_system_ratio=0.0,
        empty_think_ratio=0.2,
    )
    assert len(dataset) == 2
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert set(sample.keys()) == {"input_ids", "labels"}
