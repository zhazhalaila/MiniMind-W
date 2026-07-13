import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch_sft.dataloader import build_sft_dataloader, collate_sft_batch  # noqa: E402
from scratch_sft.dataset import SFTDataset, build_sft_example, load_sft_jsonl_records  # noqa: E402
from scratch_sft.prompt import (  # noqa: E402
    build_sft_chat_prompt,
    build_sft_special_token_ids,
    format_sft_messages,
    generate_sft_labels,
    pad_sft_example,
    postprocess_sft_prompt,
)
from scratch_sft.train_loop import compute_sft_loss, run_sft_train_loop, train_sft_one_step  # noqa: E402


class FakeTokenizer:
    bos_token = "<bos>"
    eos_token = "<eos>"
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None):
        del tokenize
        del tools
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        prompt = "\n".join(parts)
        if add_generation_prompt:
            prompt += "\nassistant: "
        return prompt

    def __call__(self, text, add_special_tokens=False, max_length=None, truncation=False):
        del add_special_tokens
        token_ids = [ord(ch) % 50 + 10 for ch in text]
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        return type("Tokenized", (), {"input_ids": token_ids})()


class ToyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        del idx
        return {
            "input_ids": torch.tensor([1, 2, 3, 0], dtype=torch.long),
            "labels": torch.tensor([-100, -100, 3, 0], dtype=torch.long),
        }


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, input_ids, labels):
        del input_ids, labels
        return type("Output", (), {"loss": self.weight.sum()})()


class SFTChecks(unittest.TestCase):
    def _call_or_skip(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            self.skipTest(str(exc))

    def test_format_sft_messages(self):
        record = {
            "conversations": [
                {"role": "user", "content": "Introduce Hangzhou."},
                {"role": "assistant", "content": "Hangzhou is the capital of Zhejiang."},
            ]
        }
        actual = self._call_or_skip(
            format_sft_messages,
            record=record,
            add_system_ratio=0.0,
        )
        self.assertIsInstance(actual, list)
        self.assertTrue(all(isinstance(item, dict) for item in actual))

    def test_build_sft_chat_prompt(self):
        tokenizer = FakeTokenizer()
        messages = [
            {"role": "user", "content": "Introduce Hangzhou."},
            {"role": "assistant", "content": "Hangzhou is the capital of Zhejiang."},
        ]
        actual = self._call_or_skip(build_sft_chat_prompt, messages, tokenizer)
        self.assertIsInstance(actual, str)

    def test_postprocess_sft_prompt(self):
        actual = self._call_or_skip(
            postprocess_sft_prompt,
            "<think>\n\n</think>\n\nassistant: hello",
            0.2,
        )
        self.assertIsInstance(actual, str)

    def test_build_sft_special_token_ids(self):
        bos_ids, eos_ids = self._call_or_skip(build_sft_special_token_ids, FakeTokenizer())
        self.assertIsInstance(bos_ids, list)
        self.assertIsInstance(eos_ids, list)

    def test_generate_sft_labels(self):
        labels = self._call_or_skip(
            generate_sft_labels,
            [11, 12, 13, 14, 15, 16],
            [11, 12],
            [15, 16],
            6,
        )
        self.assertIsInstance(labels, list)
        self.assertEqual(len(labels), 6)

    def test_pad_sft_example(self):
        padded_input_ids, padded_labels = self._call_or_skip(
            pad_sft_example,
            [11, 12, 13],
            [-100, -100, 13],
            0,
            8,
        )
        self.assertEqual(len(padded_input_ids), 8)
        self.assertEqual(len(padded_labels), 8)

    def test_load_sft_jsonl_records(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "toy_sft.jsonl"
            rows = [
                {"conversations": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
                {"conversations": [{"role": "user", "content": "where"}, {"role": "assistant", "content": "Zhejiang"}]},
            ]
            with path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            actual = self._call_or_skip(load_sft_jsonl_records, str(path))
            self.assertIsInstance(actual, list)
            self.assertEqual(len(actual), 2)

    def test_build_sft_example(self):
        tokenizer = FakeTokenizer()
        record = {
            "conversations": [
                {"role": "user", "content": "Introduce Hangzhou."},
                {"role": "assistant", "content": "Hangzhou is the capital of Zhejiang."},
            ],
        }
        assistant_bos_ids, assistant_eos_ids = self._call_or_skip(build_sft_special_token_ids, tokenizer)
        actual = self._call_or_skip(
            build_sft_example,
            record,
            tokenizer,
            assistant_bos_ids,
            assistant_eos_ids,
            24,
            0.0,
            0.2,
        )
        self.assertIsInstance(actual, dict)
        self.assertEqual(set(actual.keys()), {"input_ids", "labels"})
        self.assertEqual(actual["input_ids"].ndim, 1)
        self.assertEqual(actual["labels"].ndim, 1)

    def test_sft_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "toy_sft.jsonl"
            rows = [
                {"conversations": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
                {"conversations": [{"role": "user", "content": "where"}, {"role": "assistant", "content": "Zhejiang"}]},
            ]
            with path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            dataset = self._call_or_skip(
                SFTDataset,
                str(path),
                FakeTokenizer(),
                24,
                0.0,
                0.2,
            )
            self.assertEqual(len(dataset), 2)
            sample = dataset[0]
            self.assertIsInstance(sample, dict)

    def test_collate_sft_batch(self):
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
        actual = self._call_or_skip(collate_sft_batch, examples)
        self.assertEqual(actual["input_ids"].shape, (2, 3))
        self.assertEqual(actual["labels"].shape, (2, 3))

    def test_build_sft_dataloader(self):
        actual = self._call_or_skip(build_sft_dataloader, ToyDataset(), 2, False)
        batch = next(iter(actual))
        self.assertEqual(batch["input_ids"].shape, (2, 4))
        self.assertEqual(batch["labels"].shape, (2, 4))

    def test_compute_sft_loss(self):
        model = ToyModel()
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "labels": torch.tensor([[-100, -100, 3]], dtype=torch.long),
        }
        actual = self._call_or_skip(compute_sft_loss, model, batch)
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.ndim, 0)

    def test_train_sft_one_step(self):
        model = ToyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "labels": torch.tensor([[-100, -100, 3]], dtype=torch.long),
        }
        actual = self._call_or_skip(train_sft_one_step, model, batch, optimizer, "cpu")
        self.assertIsInstance(actual, float)

    def test_run_sft_train_loop(self):
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
        actual = self._call_or_skip(
            run_sft_train_loop,
            model,
            dataloader,
            optimizer,
            "cpu",
            2,
            1,
            None,
            None,
        )
        self.assertIsInstance(actual, list)
        self.assertEqual(len(actual), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
