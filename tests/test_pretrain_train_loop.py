import json

import pytest
import torch

from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from scratch_pretrain.checkpoint import build_checkpoint_state, save_checkpoint
from scratch_pretrain.config import build_pretrain_train_config
from scratch_pretrain.dataloader import build_pretrain_dataloader, collate_pretrain_batch
from scratch_pretrain.optim import build_optimizer
from scratch_pretrain.train_loop import (
    compute_pretrain_loss,
    move_batch_to_device,
    run_pretrain_train_loop,
    train_one_step,
)


def xfail_on_not_implemented(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.xfail(f"{fn.__name__} is not implemented yet.")


def build_tiny_model():
    config = MiniMindConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
    )
    return MiniMindForCausalLM(config)


def build_toy_batch(batch_size=2, seq_len=8):
    input_ids = torch.randint(0, 32, (batch_size, seq_len), dtype=torch.long)
    labels = input_ids.clone()
    labels[:, -2:] = -100
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def test_build_pretrain_train_config_fields():
    actual = xfail_on_not_implemented(
        build_pretrain_train_config,
        "checkpoints",
        2,
        1e-3,
        0.01,
        5,
        "cpu",
        1,
        2,
    )
    assert actual.save_dir == "checkpoints"
    assert actual.batch_size == 2
    assert actual.learning_rate == pytest.approx(1e-3)
    assert actual.weight_decay == pytest.approx(0.01)
    assert actual.max_steps == 5
    assert actual.device == "cpu"
    assert actual.log_every == 1
    assert actual.save_every == 2


def test_collate_pretrain_batch_stacks_tensors():
    examples = [
        (torch.arange(8, dtype=torch.long), torch.arange(8, dtype=torch.long)),
        (torch.arange(8, dtype=torch.long) + 1, torch.arange(8, dtype=torch.long) + 1),
    ]
    batch = xfail_on_not_implemented(collate_pretrain_batch, examples)
    assert isinstance(batch, dict)
    assert set(batch.keys()) == {"input_ids", "labels"}
    assert batch["input_ids"].shape == (2, 8)
    assert batch["labels"].shape == (2, 8)


def test_build_pretrain_dataloader_returns_batched_dict(tmp_path, fake_tokenizer):
    data_path = tmp_path / "toy.jsonl"
    rows = [{"text": "hello"}, {"text": "world"}, {"text": "mini"}]
    with data_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    dataset = PretrainDataset(str(data_path), fake_tokenizer, max_length=8)
    dataloader = xfail_on_not_implemented(build_pretrain_dataloader, dataset, 2, False)
    batch = next(iter(dataloader))

    assert isinstance(batch, dict)
    assert batch["input_ids"].shape == (2, 8)
    assert batch["labels"].shape == (2, 8)


def test_build_optimizer_returns_optimizer():
    model = build_tiny_model()
    actual = xfail_on_not_implemented(build_optimizer, model, 1e-3, 0.01)
    assert isinstance(actual, torch.optim.Optimizer)


def test_build_checkpoint_state_contains_expected_keys():
    model = build_tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    actual = xfail_on_not_implemented(build_checkpoint_state, model, optimizer, 3)
    assert isinstance(actual, dict)
    assert set(actual.keys()) == {"model", "optimizer", "step"}
    assert actual["step"] == 3


def test_save_checkpoint_writes_file(tmp_path):
    checkpoint = {
        "model": {"layer.weight": torch.ones(1)},
        "optimizer": {"state": {}, "param_groups": []},
        "step": 2,
    }
    save_path = tmp_path / "toy_checkpoint.pt"
    xfail_on_not_implemented(save_checkpoint, checkpoint, str(save_path))
    assert save_path.exists()


def test_move_batch_to_device_keeps_shapes():
    batch = build_toy_batch()
    actual = xfail_on_not_implemented(move_batch_to_device, batch, torch.device("cpu"))
    assert set(actual.keys()) == {"input_ids", "labels"}
    assert actual["input_ids"].shape == (2, 8)
    assert actual["labels"].shape == (2, 8)
    assert actual["input_ids"].device.type == "cpu"
    assert actual["labels"].device.type == "cpu"


def test_compute_pretrain_loss_returns_scalar():
    model = build_tiny_model()
    batch = build_toy_batch()
    actual = xfail_on_not_implemented(compute_pretrain_loss, model, batch)
    assert isinstance(actual, torch.Tensor)
    assert actual.ndim == 0


def test_train_one_step_returns_python_float():
    model = build_tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = build_toy_batch()
    actual = xfail_on_not_implemented(train_one_step, model, batch, optimizer, torch.device("cpu"))
    assert isinstance(actual, float)


def test_run_pretrain_train_loop_returns_loss_history():
    model = build_tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataloader = [build_toy_batch(), build_toy_batch()]
    actual = xfail_on_not_implemented(
        run_pretrain_train_loop,
        model,
        dataloader,
        optimizer,
        torch.device("cpu"),
        2,
        1,
        None,
        None,
    )
    assert isinstance(actual, list)
    assert len(actual) == 2
    assert all(isinstance(loss, float) for loss in actual)
