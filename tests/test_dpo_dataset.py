import json

import pytest
import torch

from scratch_dpo.dataset import (
    DPODataset,
    build_dpo_pair_example,
    build_dpo_sequence_tensors,
    build_dpo_special_token_ids,
    generate_dpo_loss_mask,
    load_dpo_jsonl_records,
)


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
        del add_generation_prompt
        del tools
        return "\n".join(f"{item['role']}: {item['content']}" for item in messages)

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None, padding=None):
        del add_special_tokens
        token_ids = [ord(ch) % 50 + 10 for ch in text]
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        if padding == "max_length" and max_length is not None:
            token_ids = token_ids + [self.pad_token_id] * max(0, max_length - len(token_ids))
        return {"input_ids": token_ids}


def test_load_dpo_jsonl_records_reads_chosen_rejected(tmp_path):
    path = tmp_path / "toy_dpo.jsonl"
    row = {
        "chosen": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        "rejected": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "bad"}],
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    records = xfail_on_not_implemented(load_dpo_jsonl_records, str(path))
    assert len(records) == 1


def test_build_dpo_sequence_tensors_returns_shifted_rank1_tensors():
    x, y, mask = xfail_on_not_implemented(
        build_dpo_sequence_tensors,
        [1, 2, 3],
        [0, 1, 1],
        0,
        6,
    )
    assert isinstance(x, torch.Tensor)
    assert x.shape == y.shape == mask.shape == (5,)


def test_build_dpo_pair_example_has_minimind_keys():
    tokenizer = FakeTokenizer()
    bos_ids, eos_ids = xfail_on_not_implemented(build_dpo_special_token_ids, tokenizer)
    record = {
        "chosen": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        "rejected": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "bad"}],
    }
    sample = xfail_on_not_implemented(build_dpo_pair_example, record, tokenizer, bos_ids, eos_ids, 16, 0.2)
    assert set(sample.keys()) == {
        "x_chosen",
        "y_chosen",
        "mask_chosen",
        "x_rejected",
        "y_rejected",
        "mask_rejected",
    }


def test_generate_dpo_loss_mask_returns_list():
    mask = xfail_on_not_implemented(generate_dpo_loss_mask, [1, 2, 3], [1], [3], 8)
    assert isinstance(mask, list)


def test_dpo_dataset_returns_single_pair(tmp_path):
    path = tmp_path / "toy_dpo.jsonl"
    row = {
        "chosen": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        "rejected": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "bad"}],
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    dataset = xfail_on_not_implemented(DPODataset, str(path), FakeTokenizer(), 16, 0.2)
    sample = dataset[0]
    assert "x_chosen" in sample

