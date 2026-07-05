import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402
from scratch_pretrain.config import build_pretrain_data_config, build_pretrain_train_config  # noqa: E402
from scratch_pretrain.entry import (  # noqa: E402
    build_model,
    build_pretrain_runtime,
    build_smoke_test_configs,
    load_checkpoint_file,
    main,
    run_pretrain_entry,
    run_pretrain_smoke_test,
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


class PretrainEntryChecks(unittest.TestCase):
    def build_tiny_model_config(self):
        return MiniMindConfig(
            vocab_size=128,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=32,
        )

    def _call_or_skip(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            self.skipTest(str(exc))

    def test_build_model(self):
        actual = self._call_or_skip(
            build_model,
            self.build_tiny_model_config(),
            torch.device("cpu"),
        )
        self.assertIsInstance(actual, MiniMindForCausalLM)
        self.assertEqual(next(actual.parameters()).device.type, "cpu")

    def test_build_pretrain_runtime(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "toy.jsonl"
            rows = [{"text": "hello"}, {"text": "world"}, {"text": "mini"}]
            with path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            data_config = build_pretrain_data_config(str(Path(tmp_dir) / "tokenizer"), str(path), 8)
            train_config = build_pretrain_train_config(str(Path(tmp_dir) / "ckpt"), 2, 1e-3, 0.01, 2, "cpu", 1, 2)

            with patch("scratch_pretrain.entry.load_tokenizer", return_value=FakeTokenizer()):
                actual = self._call_or_skip(
                    build_pretrain_runtime,
                    data_config,
                    train_config,
                    self.build_tiny_model_config(),
                )

            self.assertIsInstance(actual, dict)
            self.assertEqual(
                set(actual.keys()),
                {"tokenizer", "dataset", "dataloader", "model", "optimizer"},
            )

    def test_run_pretrain_entry(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "toy.jsonl"
            rows = [{"text": "hello"}, {"text": "world"}, {"text": "mini"}]
            with path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            data_config = build_pretrain_data_config(str(Path(tmp_dir) / "tokenizer"), str(path), 8)
            train_config = build_pretrain_train_config(str(Path(tmp_dir) / "ckpt"), 2, 1e-3, 0.01, 2, "cpu", 1, 2)

            with patch("scratch_pretrain.entry.load_tokenizer", return_value=FakeTokenizer()):
                actual = self._call_or_skip(
                    run_pretrain_entry,
                    data_config,
                    train_config,
                    self.build_tiny_model_config(),
                )

            self.assertIsInstance(actual, list)
            self.assertEqual(len(actual), 2)
            self.assertTrue(all(isinstance(loss, float) for loss in actual))

    def test_load_checkpoint_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "toy_checkpoint.pt"
            checkpoint = {
                "model": {"layer.weight": torch.ones(1)},
                "optimizer": {"state": {}, "param_groups": []},
                "step": 2,
            }
            torch.save(checkpoint, path)

            actual = self._call_or_skip(load_checkpoint_file, str(path), torch.device("cpu"))
            self.assertIsInstance(actual, dict)
            self.assertEqual(set(actual.keys()), {"model", "optimizer", "step"})
            self.assertEqual(actual["step"], 2)

    def test_build_smoke_test_configs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_config, train_config, model_config = self._call_or_skip(
                build_smoke_test_configs,
                tmp_dir,
                "cpu",
            )
            self.assertEqual(data_config.data_path, str(Path(tmp_dir) / "data" / "pretrain_t2t_mini.jsonl"))
            self.assertEqual(data_config.tokenizer_dir, str(Path(tmp_dir) / "tokenizer"))
            self.assertEqual(train_config.device, "cpu")
            self.assertIsInstance(model_config, MiniMindConfig)

    def test_run_pretrain_smoke_test(self):
        fake_data_config = build_pretrain_data_config("tokenizer", "data.jsonl", 8)
        fake_train_config = build_pretrain_train_config("checkpoints", 2, 1e-3, 0.01, 2, "cpu", 1, 2)
        fake_model_config = self.build_tiny_model_config()

        with patch(
            "scratch_pretrain.entry.build_smoke_test_configs",
            return_value=(fake_data_config, fake_train_config, fake_model_config),
        ), patch(
            "scratch_pretrain.entry.run_pretrain_entry",
            return_value=[1.0, 0.9],
        ):
            actual = self._call_or_skip(run_pretrain_smoke_test, "/tmp/project", "cpu")

        self.assertEqual(actual, [1.0, 0.9])

    def test_main(self):
        with patch("scratch_pretrain.entry.run_pretrain_smoke_test", return_value=[1.0, 0.9]):
            self._call_or_skip(main)


if __name__ == "__main__":
    unittest.main(verbosity=2)
