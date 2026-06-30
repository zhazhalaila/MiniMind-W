import json

import pytest
import torch

from dataset.lm_dataset import PretrainDataset
from scratch_pretrain.dataset import build_pretrain_example, load_jsonl_records


def xfail_on_not_implemented(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.xfail(f"{fn.__name__} is not implemented yet.")


def test_load_jsonl_records_reads_all_lines(tmp_path):
    data_path = tmp_path / "toy.jsonl"
    rows = [{"text": "hello"}, {"text": "world"}]
    with data_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    actual = xfail_on_not_implemented(load_jsonl_records, str(data_path))
    assert actual == rows


def test_build_pretrain_example_returns_tensors(fake_tokenizer, sample_record):
    input_ids, labels = xfail_on_not_implemented(
        build_pretrain_example,
        sample_record,
        fake_tokenizer,
        8,
    )
    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert input_ids.dtype == torch.long
    assert labels.dtype == torch.long
    assert input_ids.shape == (8,)
    assert labels.shape == (8,)


def test_build_pretrain_example_masks_pad_positions(fake_tokenizer, sample_record):
    input_ids, labels = xfail_on_not_implemented(
        build_pretrain_example,
        sample_record,
        fake_tokenizer,
        10,
    )
    pad_positions = input_ids == fake_tokenizer.pad_token_id
    assert torch.equal(labels[pad_positions], torch.full_like(labels[pad_positions], -100))


def test_pretrain_dataset_len_and_getitem(tmp_path, fake_tokenizer):
    data_path = tmp_path / "toy.jsonl"
    rows = [{"text": "hello"}, {"text": "world"}, {"text": "mini"}]
    with data_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    try:
        ds = PretrainDataset(str(data_path), fake_tokenizer, max_length=8)
        length = len(ds)
        item = ds[0]
    except NotImplementedError as exc:
        pytest.xfail(str(exc))

    assert length == 3
    assert isinstance(item, tuple)
    assert len(item) == 2
    assert item[0].shape == (8,)
    assert item[1].shape == (8,)
