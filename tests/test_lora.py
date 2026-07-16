import pytest
import torch

from scratch_lora.lora import LoRA, apply_lora, mark_only_lora_as_trainable, parse_target_modules
from scratch_lora.train_lora import build_lora_parser


def xfail_on_not_implemented(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.xfail(f"{fn.__name__} is not implemented yet.")


class ToyLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False)


def test_lora_forward_shape():
    lora = xfail_on_not_implemented(LoRA, 4, 6, 2)
    y = xfail_on_not_implemented(lora.forward, torch.randn(3, 4))
    assert y.shape == (3, 6)


def test_parse_target_modules():
    actual = xfail_on_not_implemented(parse_target_modules, "q_proj,k_proj")
    assert actual == ["q_proj", "k_proj"]


def test_apply_lora_and_freeze_base():
    model = ToyLinearModel()
    model = xfail_on_not_implemented(apply_lora, model, 2, None, True)
    params = xfail_on_not_implemented(mark_only_lora_as_trainable, model)
    assert isinstance(params, list)


def test_lora_parser_defaults():
    args = build_lora_parser().parse_args([])
    assert args.lora_name == "lora_medical"
    assert args.rank == 16

