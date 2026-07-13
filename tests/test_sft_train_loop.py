from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scratch_sft.dataloader import build_sft_dataloader, collate_sft_batch
from scratch_sft.train_loop import compute_sft_loss, run_sft_train_loop, train_sft_one_step


def xfail_on_not_implemented(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.xfail(f"{fn.__name__} is not implemented yet.")


class ToyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        del idx
        return {
            "input_ids": torch.tensor([1, 2, 3, 0], dtype=torch.long),
            "labels": torch.tensor([-100, -100, 3, 0], dtype=torch.long),
        }


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, input_ids, labels):
        del input_ids, labels
        return SimpleNamespace(loss=self.weight.sum())


def test_collate_sft_batch_returns_rank2_tensors():
    examples = [
        {
            "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "labels": torch.tensor([-100, -100, 3], dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([4, 5, 6], dtype=torch.long),
            "labels": torch.tensor([-100, 5, 6], dtype=torch.long),
        },
    ]
    actual = xfail_on_not_implemented(collate_sft_batch, examples)
    assert isinstance(actual, dict)
    assert actual["input_ids"].shape == (2, 3)
    assert actual["labels"].shape == (2, 3)


def test_build_sft_dataloader_returns_batches():
    dataset = ToyDataset()
    actual = xfail_on_not_implemented(build_sft_dataloader, dataset, batch_size=2, shuffle=False)
    batch = next(iter(actual))
    assert batch["input_ids"].shape == (2, 4)
    assert batch["labels"].shape == (2, 4)


def test_compute_sft_loss_returns_scalar():
    model = ToyModel()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 3]], dtype=torch.long),
    }
    actual = xfail_on_not_implemented(compute_sft_loss, model, batch)
    assert isinstance(actual, torch.Tensor)
    assert actual.ndim == 0


def test_train_sft_one_step_returns_float():
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 3]], dtype=torch.long),
    }
    actual = xfail_on_not_implemented(train_sft_one_step, model, batch, optimizer, "cpu")
    assert isinstance(actual, float)


def test_run_sft_train_loop_returns_loss_history():
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataloader = [
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "labels": torch.tensor([[-100, -100, 3]], dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([[4, 5, 6]], dtype=torch.long),
            "labels": torch.tensor([[-100, 5, 6]], dtype=torch.long),
        },
    ]
    actual = xfail_on_not_implemented(
        run_sft_train_loop,
        model,
        dataloader,
        optimizer,
        "cpu",
        max_steps=2,
        log_every=1,
        save_every=None,
        save_dir=None,
    )
    assert isinstance(actual, list)
    assert len(actual) == 2
    assert all(isinstance(loss, float) for loss in actual)
