import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model_minimind import MiniMindConfig  # noqa: E402
from scratch_pretrain.config import PretrainDataConfig, PretrainTrainConfig  # noqa: E402
from scratch_pretrain.train_pretrain import (  # noqa: E402
    build_autocast_context,
    append_train_metric,
    build_data_config_from_args,
    build_grad_scaler,
    build_model_config_from_args,
    build_runtime_from_args,
    build_train_config_from_args,
    build_train_parser,
    compute_learning_rate,
    format_train_log,
    load_training_state,
    main,
    parse_train_args,
    run_formal_pretrain,
)


class DummyTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __len__(self):
        return 6400


class DummyOutput:
    def __init__(self, loss: torch.Tensor):
        self.loss = loss


class TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor) -> DummyOutput:
        valid_mask = labels.ne(-100).float()
        target = torch.where(labels.eq(-100), torch.zeros_like(labels), labels)
        diff = (input_ids.float() - target.float()) * valid_mask
        loss = diff.pow(2).mean() + self.scale.pow(2)
        return DummyOutput(loss)


class PretrainFormalTrainChecks(unittest.TestCase):
    def build_fake_args(self):
        return Namespace(
            save_dir="checkpoints",
            log_dir="logs",
            checkpoint_dir="checkpoints",
            out_dir="out",
            save_weight="pretrain",
            epochs=2,
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            device="cpu",
            dtype="float32",
            num_workers=0,
            accumulation_steps=1,
            grad_clip=1.0,
            log_interval=1,
            save_interval=2,
            warmup_steps=0,
            min_lr_ratio=0.1,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=32,
            max_seq_len=8,
            tokenizer_dir="tokenizer",
            data_path="data/pretrain_t2t_mini.jsonl",
            from_weight="none",
            from_resume="none",
        )

    def _call_or_skip(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            self.skipTest(str(exc))

    def test_build_train_parser(self):
        parser = self._call_or_skip(build_train_parser)
        args = parser.parse_args(
            [
                "--checkpoint_dir",
                "checkpoints",
                "--log_dir",
                "logs",
                "--out_dir",
                "out",
                "--tokenizer_dir",
                "tokenizer",
                "--data_path",
                "data/pretrain_t2t_mini.jsonl",
            ]
        )
        self.assertTrue(hasattr(args, "checkpoint_dir"))
        self.assertTrue(hasattr(args, "log_dir"))
        self.assertTrue(hasattr(args, "out_dir"))
        self.assertTrue(hasattr(args, "tokenizer_dir"))
        self.assertTrue(hasattr(args, "data_path"))

    def test_parse_train_args(self):
        actual = self._call_or_skip(
            parse_train_args,
            [
                "--checkpoint_dir",
                "checkpoints",
                "--log_dir",
                "logs",
                "--out_dir",
                "out",
                "--tokenizer_dir",
                "tokenizer",
                "--data_path",
                "data/pretrain_t2t_mini.jsonl",
            ],
        )
        self.assertIsInstance(actual, Namespace)
        self.assertEqual(actual.checkpoint_dir, "checkpoints")
        self.assertEqual(actual.log_dir, "logs")
        self.assertEqual(actual.out_dir, "out")
        self.assertEqual(actual.tokenizer_dir, "tokenizer")

    def test_parse_train_args_rejects_invalid_accumulation_steps(self):
        with self.assertRaises(SystemExit):
            parse_train_args(
                [
                    "--checkpoint_dir",
                    "checkpoints",
                    "--log_dir",
                    "logs",
                    "--out_dir",
                    "out",
                    "--tokenizer_dir",
                    "tokenizer",
                    "--data_path",
                    "data/pretrain_t2t_mini.jsonl",
                    "--accumulation_steps",
                    "0",
                ],
            )

    def test_build_data_config_from_args(self):
        actual = self._call_or_skip(build_data_config_from_args, self.build_fake_args())
        self.assertIsInstance(actual, PretrainDataConfig)
        self.assertEqual(actual.tokenizer_dir, "tokenizer")
        self.assertEqual(actual.data_path, "data/pretrain_t2t_mini.jsonl")

    def test_build_train_config_from_args(self):
        actual = self._call_or_skip(build_train_config_from_args, self.build_fake_args())
        self.assertIsInstance(actual, PretrainTrainConfig)
        self.assertEqual(actual.save_dir, "checkpoints")
        self.assertEqual(actual.batch_size, 2)

    def test_build_model_config_from_args(self):
        actual = self._call_or_skip(build_model_config_from_args, self.build_fake_args())
        self.assertIsInstance(actual, MiniMindConfig)
        self.assertEqual(actual.hidden_size, 16)
        self.assertEqual(actual.num_hidden_layers, 2)

    def test_build_runtime_from_args(self):
        args = self.build_fake_args()
        args.num_workers = 3
        with patch("scratch_pretrain.train_pretrain.load_tokenizer", return_value=DummyTokenizer()):
            actual = self._call_or_skip(build_runtime_from_args, args)
        self.assertIsInstance(actual, dict)
        self.assertEqual(
            set(actual.keys()),
            {"tokenizer", "dataset", "dataloader", "model", "optimizer"},
        )
        self.assertEqual(actual["dataloader"].num_workers, 3)

    def test_build_autocast_context(self):
        actual = self._call_or_skip(build_autocast_context, "cpu", "float32")
        self.assertIsNotNone(actual)

    def test_build_grad_scaler(self):
        actual = self._call_or_skip(build_grad_scaler, "cpu", "float32")
        self.assertFalse(actual.is_enabled())

    def test_compute_learning_rate(self):
        first = self._call_or_skip(compute_learning_rate, 1, 4, 1e-3, 0, 0.1)
        last = self._call_or_skip(compute_learning_rate, 4, 4, 1e-3, 0, 0.1)
        warmup = self._call_or_skip(compute_learning_rate, 1, 8, 1e-3, 2, 0.1)
        self.assertAlmostEqual(first, 1e-3)
        self.assertAlmostEqual(last, 1e-4)
        self.assertAlmostEqual(warmup, 5e-4)

    def test_format_train_log(self):
        actual = self._call_or_skip(format_train_log, 3, 1.2345, 1e-3)
        self.assertIsInstance(actual, str)
        self.assertIn("3", actual)

    def test_append_train_metric(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics_path = Path(tmp_dir) / "train_metrics.jsonl"
            self._call_or_skip(append_train_metric, str(metrics_path), 3, 1.2345, 1e-3)
            self.assertTrue(metrics_path.exists())
            rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["step"], 3)
            self.assertAlmostEqual(rows[0]["loss"], 1.2345)
            self.assertAlmostEqual(rows[0]["learning_rate"], 1e-3)

    def test_load_training_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "resume.pt"
            model = torch.nn.Linear(4, 4)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": 7,
                },
                checkpoint_path,
            )
            actual = self._call_or_skip(
                load_training_state,
                str(checkpoint_path),
                model,
                optimizer,
                "cpu",
            )
            self.assertEqual(actual, 7)

    def test_run_formal_pretrain(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self.build_fake_args()
            args.log_dir = str(Path(tmp_dir) / "logs")
            args.checkpoint_dir = str(Path(tmp_dir) / "checkpoints")
            args.out_dir = str(Path(tmp_dir) / "out")
            args.save_weight = "toy_pretrain"
            args.epochs = 2
            args.save_interval = 1
            args.log_interval = 1
            args.accumulation_steps = 1

            batch = {
                "input_ids": torch.tensor([[1, 2, 3, 0]], dtype=torch.long),
                "labels": torch.tensor([[1, 2, 3, -100]], dtype=torch.long),
            }
            model = TinyCausalLM()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

            runtime = {
                "tokenizer": DummyTokenizer(),
                "dataset": object(),
                "dataloader": [batch],
                "model": model,
                "optimizer": optimizer,
            }

            with patch(
                "scratch_pretrain.train_pretrain.build_runtime_from_args",
                return_value=runtime,
            ):
                actual = self._call_or_skip(run_formal_pretrain, args)

            self.assertEqual(len(actual), 2)
            self.assertTrue(all(isinstance(value, float) for value in actual))

            metrics_path = Path(args.log_dir) / f"{args.save_weight}_metrics.jsonl"
            text_log_path = Path(args.log_dir) / f"{args.save_weight}.log"
            resume_checkpoint_path = Path(args.checkpoint_dir) / f"{args.save_weight}_resume_latest.pt"
            final_weight_path = Path(args.out_dir) / f"{args.save_weight}_final.pt"

            self.assertTrue(metrics_path.exists())
            self.assertTrue(text_log_path.exists())
            self.assertTrue(resume_checkpoint_path.exists())
            self.assertFalse((Path(args.checkpoint_dir) / "step_1.pt").exists())
            self.assertFalse((Path(args.checkpoint_dir) / "step_2.pt").exists())
            self.assertTrue(final_weight_path.exists())

            rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["step"], 1)
            self.assertEqual(rows[1]["step"], 2)
            self.assertAlmostEqual(rows[0]["learning_rate"], args.learning_rate)
            self.assertLess(rows[1]["learning_rate"], rows[0]["learning_rate"])

    def test_run_formal_pretrain_resume(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self.build_fake_args()
            args.log_dir = str(Path(tmp_dir) / "logs")
            args.checkpoint_dir = str(Path(tmp_dir) / "checkpoints")
            args.out_dir = str(Path(tmp_dir) / "out")
            args.save_weight = "resume_case"
            args.epochs = 2
            args.save_interval = 10
            args.log_interval = 1
            args.accumulation_steps = 1

            model = TinyCausalLM()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

            resume_path = Path(tmp_dir) / "resume.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": 1,
                    "epoch": 0,
                    "batch_in_epoch": 1,
                },
                resume_path,
            )
            args.from_resume = str(resume_path)

            batch_a = {
                "input_ids": torch.tensor([[1, 2, 3, 0]], dtype=torch.long),
                "labels": torch.tensor([[1, 2, 3, -100]], dtype=torch.long),
            }
            batch_b = {
                "input_ids": torch.tensor([[3, 2, 1, 0]], dtype=torch.long),
                "labels": torch.tensor([[3, 2, 1, -100]], dtype=torch.long),
            }

            runtime = {
                "tokenizer": DummyTokenizer(),
                "dataset": object(),
                "dataloader": [batch_a, batch_b],
                "model": model,
                "optimizer": optimizer,
            }

            with patch(
                "scratch_pretrain.train_pretrain.build_runtime_from_args",
                return_value=runtime,
            ):
                actual = self._call_or_skip(run_formal_pretrain, args)

            self.assertEqual(len(actual), 3)

            resume_checkpoint_path = Path(args.checkpoint_dir) / f"{args.save_weight}_resume_latest.pt"
            checkpoint = torch.load(resume_checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["step"], 4)
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertEqual(checkpoint["batch_in_epoch"], 0)
            self.assertIn("scaler", checkpoint)

    def test_main(self):
        with patch(
            "scratch_pretrain.train_pretrain.parse_train_args",
            return_value=self.build_fake_args(),
        ), patch(
            "scratch_pretrain.train_pretrain.run_formal_pretrain",
            return_value=[1.0, 0.9],
        ):
            self._call_or_skip(main)


if __name__ == "__main__":
    unittest.main(verbosity=2)
