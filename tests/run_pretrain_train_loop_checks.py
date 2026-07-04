import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.lm_dataset import PretrainDataset  # noqa: E402
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402
from scratch_pretrain.checkpoint import build_checkpoint_state, save_checkpoint  # noqa: E402
from scratch_pretrain.config import build_pretrain_train_config  # noqa: E402
from scratch_pretrain.dataloader import build_pretrain_dataloader, collate_pretrain_batch  # noqa: E402
from scratch_pretrain.optim import build_optimizer  # noqa: E402
from scratch_pretrain.train_loop import (  # noqa: E402
    compute_pretrain_loss,
    move_batch_to_device,
    run_pretrain_train_loop,
    train_one_step,
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


class PretrainTrainLoopChecks(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.config = MiniMindConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=32,
        )
        self.model = MiniMindForCausalLM(self.config)
        self.batch = {
            "input_ids": torch.randint(0, 32, (2, 8), dtype=torch.long),
            "labels": torch.randint(0, 32, (2, 8), dtype=torch.long),
        }
        self.batch["labels"][:, -2:] = -100

    def _call_or_skip(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            self.skipTest(str(exc))

    def test_build_pretrain_train_config(self):
        actual = self._call_or_skip(
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
        self.assertEqual(actual.save_dir, "checkpoints")
        self.assertEqual(actual.batch_size, 2)
        self.assertEqual(actual.max_steps, 5)
        self.assertEqual(actual.device, "cpu")

    def test_collate_pretrain_batch(self):
        examples = [
            (torch.arange(8, dtype=torch.long), torch.arange(8, dtype=torch.long)),
            (torch.arange(8, dtype=torch.long) + 1, torch.arange(8, dtype=torch.long) + 1),
        ]
        actual = self._call_or_skip(collate_pretrain_batch, examples)
        self.assertIsInstance(actual, dict)
        self.assertEqual(set(actual.keys()), {"input_ids", "labels"})
        self.assertEqual(tuple(actual["input_ids"].shape), (2, 8))
        self.assertEqual(tuple(actual["labels"].shape), (2, 8))

    def test_build_pretrain_dataloader(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "toy.jsonl"
            rows = [{"text": "hello"}, {"text": "world"}, {"text": "mini"}]
            with path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            dataset = PretrainDataset(str(path), self.tokenizer, max_length=8)
            dataloader = self._call_or_skip(build_pretrain_dataloader, dataset, 2, False)
            batch = next(iter(dataloader))

            self.assertIsInstance(batch, dict)
            self.assertEqual(tuple(batch["input_ids"].shape), (2, 8))
            self.assertEqual(tuple(batch["labels"].shape), (2, 8))

    def test_build_optimizer(self):
        actual = self._call_or_skip(build_optimizer, self.model, 1e-3, 0.01)
        self.assertIsInstance(actual, torch.optim.Optimizer)

    def test_build_checkpoint_state(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        actual = self._call_or_skip(build_checkpoint_state, self.model, optimizer, 3)
        self.assertIsInstance(actual, dict)
        self.assertEqual(set(actual.keys()), {"model", "optimizer", "step"})
        self.assertEqual(actual["step"], 3)

    def test_save_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "toy_checkpoint.pt"
            checkpoint = {
                "model": {"layer.weight": torch.ones(1)},
                "optimizer": {"state": {}, "param_groups": []},
                "step": 2,
            }
            self._call_or_skip(save_checkpoint, checkpoint, str(path))
            self.assertTrue(path.exists())

    def test_move_batch_to_device(self):
        actual = self._call_or_skip(move_batch_to_device, self.batch, torch.device("cpu"))
        self.assertEqual(set(actual.keys()), {"input_ids", "labels"})
        self.assertEqual(tuple(actual["input_ids"].shape), (2, 8))
        self.assertEqual(tuple(actual["labels"].shape), (2, 8))
        self.assertEqual(actual["input_ids"].device.type, "cpu")

    def test_compute_pretrain_loss(self):
        actual = self._call_or_skip(compute_pretrain_loss, self.model, self.batch)
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.ndim, 0)

    def test_train_one_step(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        actual = self._call_or_skip(
            train_one_step,
            self.model,
            self.batch,
            optimizer,
            torch.device("cpu"),
        )
        self.assertIsInstance(actual, float)

    def test_run_pretrain_train_loop(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        dataloader = [self.batch, self.batch]
        actual = self._call_or_skip(
            run_pretrain_train_loop,
            self.model,
            dataloader,
            optimizer,
            torch.device("cpu"),
            2,
            1,
            None,
            None,
        )
        self.assertIsInstance(actual, list)
        self.assertEqual(len(actual), 2)
        self.assertTrue(all(isinstance(loss, float) for loss in actual))


if __name__ == "__main__":
    unittest.main(verbosity=2)
