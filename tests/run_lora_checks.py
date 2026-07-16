import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch_lora.config import build_lora_data_config, build_lora_train_config  # noqa: E402
from scratch_lora.entry import build_lora_smoke_test_configs, build_lora_runtime, run_lora_smoke_test  # noqa: E402
from scratch_lora.eval_lora import load_lora_inference_artifacts  # noqa: E402
from scratch_lora.lora import (  # noqa: E402
    LoRA,
    apply_lora,
    iter_lora_modules,
    iter_lora_parameters,
    load_lora,
    mark_only_lora_as_trainable,
    merge_lora,
    parse_target_modules,
    save_lora,
    should_apply_lora,
)
from scratch_lora.train_loop import (  # noqa: E402
    compute_lora_sft_loss,
    run_lora_train_loop,
    save_lora_training_state,
    train_lora_one_step,
)
from scratch_lora.train_lora import build_lora_parser, parse_lora_args  # noqa: E402
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402


class ToyLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False)

    def forward(self, input_ids=None, labels=None):
        del input_ids, labels
        loss = self.proj.weight.sum()
        return type("Output", (), {"loss": loss, "aux_loss": torch.zeros(())})()


class LoRAChecks(unittest.TestCase):
    def _call_or_skip(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            self.skipTest(str(exc))

    def test_build_lora_configs(self):
        data_config = self._call_or_skip(
            build_lora_data_config,
            "tokenizer",
            "data/lora_medical.jsonl",
            340,
            0.2,
            0.2,
        )
        train_config = self._call_or_skip(
            build_lora_train_config,
            "logs/lora",
            "checkpoints/lora",
            "out/lora",
            "lora_medical",
            "out/full_sft_dense/full_sft_768_final.pt",
            "none",
            10,
            32,
            1e-4,
            0.0,
            "cpu",
            "float32",
            0,
            1,
            1.0,
            10,
            1000,
            0,
            0.1,
            16,
            None,
        )
        self.assertEqual(data_config.max_seq_len, 340)
        self.assertEqual(train_config.rank, 16)

    def test_lora_module_forward_shape(self):
        lora = self._call_or_skip(LoRA, 4, 6, 2)
        x = torch.randn(3, 4)
        y = self._call_or_skip(lora.forward, x)
        self.assertEqual(y.shape, (3, 6))

    def test_parse_target_modules(self):
        actual = self._call_or_skip(parse_target_modules, "q_proj,k_proj")
        self.assertEqual(actual, ["q_proj", "k_proj"])

    def test_should_apply_lora(self):
        module = torch.nn.Linear(4, 4, bias=False)
        actual = self._call_or_skip(should_apply_lora, "proj", module, None, True)
        self.assertIsInstance(actual, bool)

    def test_apply_and_collect_lora(self):
        model = ToyLinearModel()
        model = self._call_or_skip(apply_lora, model, 2, None, True)
        modules = self._call_or_skip(iter_lora_modules, model)
        params = self._call_or_skip(iter_lora_parameters, model)
        trainable = self._call_or_skip(mark_only_lora_as_trainable, model)
        self.assertIsInstance(modules, list)
        self.assertIsInstance(params, list)
        self.assertIsInstance(trainable, list)

    def test_save_load_merge_lora(self):
        model = ToyLinearModel()
        model = self._call_or_skip(apply_lora, model, 2, None, True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            lora_path = Path(tmp_dir) / "lora.pt"
            merged_path = Path(tmp_dir) / "merged.pt"
            self._call_or_skip(save_lora, model, lora_path)
            self._call_or_skip(load_lora, model, lora_path, "cpu")
            self._call_or_skip(merge_lora, model, lora_path, merged_path, "cpu")

    def test_lora_train_loop_helpers(self):
        model = ToyLinearModel()
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "labels": torch.tensor([[-100, 2, 3]], dtype=torch.long),
        }
        loss = self._call_or_skip(compute_lora_sft_loss, model, batch)
        self.assertEqual(loss.shape, ())

    def test_train_lora_one_step(self):
        model = ToyLinearModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "labels": torch.tensor([[-100, 2, 3]], dtype=torch.long),
        }
        actual = self._call_or_skip(train_lora_one_step, model, batch, optimizer, "cpu")
        self.assertIsInstance(actual, float)

    def test_run_lora_train_loop(self):
        model = ToyLinearModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "labels": torch.tensor([[-100, 2, 3]], dtype=torch.long),
        }
        actual = self._call_or_skip(run_lora_train_loop, model, [batch], optimizer, "cpu", 1)
        self.assertEqual(len(actual), 1)

    def test_save_lora_training_state(self):
        model = ToyLinearModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "state.pt"
            self._call_or_skip(save_lora_training_state, model, optimizer, 1, path)

    def test_lora_parser(self):
        parser = build_lora_parser()
        self.assertIsInstance(parser, argparse.ArgumentParser)
        args = parse_lora_args([])
        self.assertEqual(args.lora_name, "lora_medical")
        self.assertEqual(args.rank, 16)

    def test_lora_runtime_entrypoints(self):
        data_config, train_config, model_config = self._call_or_skip(
            build_lora_smoke_test_configs,
            str(ROOT),
            "cpu",
        )
        runtime = self._call_or_skip(build_lora_runtime, data_config, train_config, model_config)
        self.assertIsInstance(runtime, dict)
        history = self._call_or_skip(run_lora_smoke_test, str(ROOT), "cpu")
        self.assertIsInstance(history, list)

    def test_load_lora_inference_artifacts(self):
        model_config = MiniMindConfig(
            vocab_size=6400,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir) / "base.pt"
            lora_path = Path(tmp_dir) / "lora.pt"

            model = MiniMindForCausalLM(model_config)
            torch.save(model.state_dict(), base_path)
            apply_lora(model, rank=2)
            save_lora(model, lora_path)

            actual = self._call_or_skip(
                load_lora_inference_artifacts,
                str(base_path),
                str(lora_path),
                str(ROOT / "tokenizer"),
                model_config,
                "cpu",
            )
        self.assertIsInstance(actual, tuple)


if __name__ == "__main__":
    unittest.main(verbosity=2)
