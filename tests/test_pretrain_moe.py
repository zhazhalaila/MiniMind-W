import argparse

import pytest
import torch

from scratch_pretrain.moe import (
    add_moe_parser_args,
    build_moe_kwargs_from_args,
    build_moe_smoke_test_kwargs,
    build_moe_weight_name,
    collect_router_aux_loss,
    combine_lm_and_router_loss,
)


def xfail_on_not_implemented(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.xfail(f"{fn.__name__} is not implemented yet.")


class DummyDenseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)


def build_fake_moe_args() -> argparse.Namespace:
    return argparse.Namespace(
        use_moe=1,
        num_experts=4,
        num_experts_per_tok=1,
        moe_intermediate_size=128,
        router_aux_loss_coef=5e-4,
    )


def test_add_moe_parser_args_registers_expected_flags():
    parser = argparse.ArgumentParser("toy_pretrain")
    parser = xfail_on_not_implemented(add_moe_parser_args, parser)
    args = parser.parse_args(
        [
            "--use_moe",
            "1",
            "--num_experts",
            "4",
            "--num_experts_per_tok",
            "1",
            "--moe_intermediate_size",
            "128",
            "--router_aux_loss_coef",
            "0.0005",
        ]
    )
    assert hasattr(args, "use_moe")
    assert hasattr(args, "num_experts")
    assert hasattr(args, "num_experts_per_tok")
    assert hasattr(args, "moe_intermediate_size")
    assert hasattr(args, "router_aux_loss_coef")


def test_build_moe_kwargs_from_args_returns_expected_keys():
    actual = xfail_on_not_implemented(build_moe_kwargs_from_args, build_fake_moe_args())
    assert isinstance(actual, dict)
    assert set(actual.keys()) == {
        "use_moe",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "router_aux_loss_coef",
    }
    assert actual["use_moe"] in (0, 1, False, True)


def test_build_moe_weight_name_supports_dense_and_moe_suffix():
    dense_name = xfail_on_not_implemented(build_moe_weight_name, "pretrain", 768, False)
    moe_name = xfail_on_not_implemented(build_moe_weight_name, "pretrain", 768, True)
    assert dense_name == "pretrain_768"
    assert moe_name == "pretrain_768_moe"


def test_collect_router_aux_loss_returns_scalar_tensor():
    model = DummyDenseModel()
    actual = xfail_on_not_implemented(collect_router_aux_loss, model)
    assert isinstance(actual, torch.Tensor)
    assert actual.shape == ()


def test_combine_lm_and_router_loss_returns_scalar_tensor():
    lm_loss = torch.tensor(1.5)
    router_aux_loss = torch.tensor(0.02)
    actual = xfail_on_not_implemented(combine_lm_and_router_loss, lm_loss, router_aux_loss)
    assert isinstance(actual, torch.Tensor)
    assert actual.shape == ()


def test_build_moe_smoke_test_kwargs_returns_tiny_moe_dict():
    actual = xfail_on_not_implemented(build_moe_smoke_test_kwargs)
    assert isinstance(actual, dict)
    assert actual["use_moe"] is True
    assert actual["num_experts"] == 2
    assert actual["num_experts_per_tok"] == 1
