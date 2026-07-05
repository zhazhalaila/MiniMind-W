import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from scratch_pretrain.config import (
    build_pretrain_data_config,
    build_pretrain_train_config,
)
from scratch_pretrain.entry import (
    build_model,
    build_pretrain_runtime,
    build_smoke_test_configs,
    load_checkpoint_file,
    main,
    run_pretrain_entry,
    run_pretrain_smoke_test,
)


def xfail_on_not_implemented(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.xfail(f"{fn.__name__} is not implemented yet.")


def build_tiny_model_config():
    return MiniMindConfig(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
    )


class FakeTokenizer:
    bos_token_id = 101
    eos_token_id = 102
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, max_length=None, truncation=False):
        token_ids = [ord(ch) % 50 + 10 for ch in text]
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        return type("Tokenized", (), {"input_ids": token_ids})()


def test_build_model_returns_causal_lm():
    model = xfail_on_not_implemented(
        build_model,
        build_tiny_model_config(),
        torch.device("cpu"),
    )
    assert isinstance(model, MiniMindForCausalLM)
    assert next(model.parameters()).device.type == "cpu"


def test_build_pretrain_runtime_contains_expected_keys(tmp_path):
    data_path = tmp_path / "toy.jsonl"
    rows = [{"text": "hello"}, {"text": "world"}, {"text": "mini"}]
    with data_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    data_config = build_pretrain_data_config(str(tmp_path / "tokenizer"), str(data_path), 8)
    train_config = build_pretrain_train_config(str(tmp_path / "ckpt"), 2, 1e-3, 0.01, 2, "cpu", 1, 2)
    model_config = build_tiny_model_config()

    with patch("scratch_pretrain.entry.load_tokenizer", return_value=FakeTokenizer()):
        runtime = xfail_on_not_implemented(
            build_pretrain_runtime,
            data_config,
            train_config,
            model_config,
        )

    assert isinstance(runtime, dict)
    assert set(runtime.keys()) == {"tokenizer", "dataset", "dataloader", "model", "optimizer"}


def test_run_pretrain_entry_returns_loss_history(tmp_path):
    data_path = tmp_path / "toy.jsonl"
    rows = [{"text": "hello"}, {"text": "world"}, {"text": "mini"}]
    with data_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    data_config = build_pretrain_data_config(str(tmp_path / "tokenizer"), str(data_path), 8)
    train_config = build_pretrain_train_config(str(tmp_path / "ckpt"), 2, 1e-3, 0.01, 2, "cpu", 1, 2)
    model_config = build_tiny_model_config()

    with patch("scratch_pretrain.entry.load_tokenizer", return_value=FakeTokenizer()):
        loss_history = xfail_on_not_implemented(
            run_pretrain_entry,
            data_config,
            train_config,
            model_config,
        )

    assert isinstance(loss_history, list)
    assert len(loss_history) == 2
    assert all(isinstance(loss, float) for loss in loss_history)


def test_load_checkpoint_file_reads_dict(tmp_path):
    checkpoint_path = tmp_path / "toy_checkpoint.pt"
    checkpoint = {
        "model": {"layer.weight": torch.ones(1)},
        "optimizer": {"state": {}, "param_groups": []},
        "step": 2,
    }
    torch.save(checkpoint, checkpoint_path)

    actual = xfail_on_not_implemented(load_checkpoint_file, str(checkpoint_path), torch.device("cpu"))
    assert isinstance(actual, dict)
    assert set(actual.keys()) == {"model", "optimizer", "step"}
    assert actual["step"] == 2


def test_build_smoke_test_configs_uses_project_root(tmp_path):
    actual = xfail_on_not_implemented(build_smoke_test_configs, str(tmp_path), "cpu")
    data_config, train_config, model_config = actual

    assert str(tmp_path / "data" / "pretrain_t2t_mini.jsonl") == data_config.data_path
    assert str(tmp_path / "tokenizer") == data_config.tokenizer_dir
    assert train_config.device == "cpu"
    assert isinstance(model_config, MiniMindConfig)


def test_run_pretrain_smoke_test_returns_loss_history(tmp_path):
    fake_data_config = build_pretrain_data_config("tokenizer", "data.jsonl", 8)
    fake_train_config = build_pretrain_train_config("checkpoints", 2, 1e-3, 0.01, 2, "cpu", 1, 2)
    fake_model_config = build_tiny_model_config()

    with patch(
        "scratch_pretrain.entry.build_smoke_test_configs",
        return_value=(fake_data_config, fake_train_config, fake_model_config),
    ), patch(
        "scratch_pretrain.entry.run_pretrain_entry",
        return_value=[1.0, 0.9],
    ):
        actual = xfail_on_not_implemented(run_pretrain_smoke_test, str(tmp_path), "cpu")

    assert actual == [1.0, 0.9]


def test_main_calls_smoke_test(tmp_path):
    with patch("scratch_pretrain.entry.run_pretrain_smoke_test", return_value=[1.0, 0.9]):
        xfail_on_not_implemented(main)
