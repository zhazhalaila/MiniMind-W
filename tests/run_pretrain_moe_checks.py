import argparse
import sys
import unittest

import torch

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch_pretrain.moe import (  # noqa: E402
    add_moe_parser_args,
    build_moe_kwargs_from_args,
    build_moe_smoke_test_kwargs,
    build_moe_weight_name,
    collect_router_aux_loss,
    combine_lm_and_router_loss,
)


class DummyDenseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)


class PretrainMoEChecks(unittest.TestCase):
    def _call_or_skip(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            self.skipTest(str(exc))

    def build_fake_moe_args(self):
        return argparse.Namespace(
            use_moe=1,
            num_experts=4,
            num_experts_per_tok=1,
            moe_intermediate_size=128,
            router_aux_loss_coef=5e-4,
        )

    def test_add_moe_parser_args(self):
        parser = argparse.ArgumentParser("toy_pretrain")
        parser = self._call_or_skip(add_moe_parser_args, parser)
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
        self.assertTrue(hasattr(args, "use_moe"))
        self.assertTrue(hasattr(args, "num_experts"))
        self.assertTrue(hasattr(args, "num_experts_per_tok"))
        self.assertTrue(hasattr(args, "moe_intermediate_size"))
        self.assertTrue(hasattr(args, "router_aux_loss_coef"))

    def test_build_moe_kwargs_from_args(self):
        actual = self._call_or_skip(build_moe_kwargs_from_args, self.build_fake_moe_args())
        self.assertIsInstance(actual, dict)
        self.assertEqual(
            set(actual.keys()),
            {
                "use_moe",
                "num_experts",
                "num_experts_per_tok",
                "moe_intermediate_size",
                "router_aux_loss_coef",
            },
        )

    def test_build_moe_weight_name(self):
        dense_name = self._call_or_skip(build_moe_weight_name, "pretrain", 768, False)
        moe_name = self._call_or_skip(build_moe_weight_name, "pretrain", 768, True)
        self.assertEqual(dense_name, "pretrain_768")
        self.assertEqual(moe_name, "pretrain_768_moe")

    def test_collect_router_aux_loss(self):
        model = DummyDenseModel()
        actual = self._call_or_skip(collect_router_aux_loss, model)
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.shape, ())

    def test_combine_lm_and_router_loss(self):
        lm_loss = torch.tensor(1.5)
        router_aux_loss = torch.tensor(0.02)
        actual = self._call_or_skip(combine_lm_and_router_loss, lm_loss, router_aux_loss)
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.shape, ())

    def test_build_moe_smoke_test_kwargs(self):
        actual = self._call_or_skip(build_moe_smoke_test_kwargs)
        self.assertIsInstance(actual, dict)
        self.assertTrue(actual["use_moe"])
        self.assertEqual(actual["num_experts"], 2)
        self.assertEqual(actual["num_experts_per_tok"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
