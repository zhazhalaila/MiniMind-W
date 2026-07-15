import pytest
import torch

from scratch_dpo.dataloader import build_dpo_dataloader, collate_dpo_batch
from scratch_dpo.train_loop import concat_chosen_rejected_batch, move_dpo_batch_to_device


def xfail_on_not_implemented(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.xfail(f"{fn.__name__} is not implemented yet.")


class ToyDPODataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        del idx
        return {
            "x_chosen": torch.tensor([1, 2, 0], dtype=torch.long),
            "y_chosen": torch.tensor([2, 3, 0], dtype=torch.long),
            "mask_chosen": torch.tensor([1, 1, 0], dtype=torch.long),
            "x_rejected": torch.tensor([1, 4, 0], dtype=torch.long),
            "y_rejected": torch.tensor([4, 5, 0], dtype=torch.long),
            "mask_rejected": torch.tensor([1, 1, 0], dtype=torch.long),
        }


def test_collate_dpo_batch_stacks_expected_keys():
    sample = ToyDPODataset()[0]
    batch = xfail_on_not_implemented(collate_dpo_batch, [sample, sample])
    assert batch["x_chosen"].shape == (2, 3)
    assert batch["x_rejected"].shape == (2, 3)


def test_build_dpo_dataloader_yields_minimind_batch():
    dataloader = xfail_on_not_implemented(build_dpo_dataloader, ToyDPODataset(), 2, False)
    batch = next(iter(dataloader))
    assert batch["x_chosen"].shape == (2, 3)


def test_concat_chosen_rejected_batch_doubles_batch_dimension():
    sample = ToyDPODataset()[0]
    batch = {key: value.unsqueeze(0) for key, value in sample.items()}
    x, y, mask = xfail_on_not_implemented(concat_chosen_rejected_batch, batch)
    assert x.shape == y.shape == mask.shape == (2, 3)


def test_move_dpo_batch_to_device_keeps_shapes():
    sample = ToyDPODataset()[0]
    batch = {key: value.unsqueeze(0) for key, value in sample.items()}
    moved = xfail_on_not_implemented(move_dpo_batch_to_device, batch, "cpu")
    assert moved["x_chosen"].shape == (1, 3)

