import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.lm_dataset import PretrainDataset  # noqa: E402
from scratch_pretrain.dataset import build_pretrain_example, load_jsonl_records  # noqa: E402
from scratch_pretrain.tokenizer_utils import load_tokenizer  # noqa: E402


class FakeTokenizer:
    bos_token_id = 101
    eos_token_id = 102
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, max_length=None, truncation=False):
        token_ids = [ord(ch) % 50 + 10 for ch in text]
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        return types.SimpleNamespace(input_ids=token_ids)


class PretrainDataChecks(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def _call_or_skip(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            self.skipTest(str(exc))

    def test_load_tokenizer(self):
        fake_tokenizer = object()
        fake_auto_tokenizer = type("FakeAutoTokenizer", (), {"from_pretrained": staticmethod(lambda path: fake_tokenizer)})
        with patch("scratch_pretrain.tokenizer_utils.AutoTokenizer", fake_auto_tokenizer):
            actual = self._call_or_skip(load_tokenizer, "tokenizer")

        self.assertIs(actual, fake_tokenizer)

    def test_load_jsonl_records(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "toy.jsonl"
            rows = [{"text": "hello"}, {"text": "world"}]
            with path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            actual = self._call_or_skip(load_jsonl_records, str(path))
            self.assertEqual(actual, rows)

    def test_build_pretrain_example(self):
        input_ids, labels = self._call_or_skip(
            build_pretrain_example,
            {"text": "hello"},
            self.tokenizer,
            8,
        )
        self.assertIsInstance(input_ids, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(input_ids.dtype, torch.long)
        self.assertEqual(labels.dtype, torch.long)
        self.assertEqual(tuple(input_ids.shape), (8,))
        self.assertEqual(tuple(labels.shape), (8,))

    def test_build_pretrain_example_masks_pad_positions(self):
        input_ids, labels = self._call_or_skip(
            build_pretrain_example,
            {"text": "hello"},
            self.tokenizer,
            10,
        )
        pad_positions = input_ids == self.tokenizer.pad_token_id
        expected = torch.full_like(labels[pad_positions], -100)
        self.assertTrue(torch.equal(labels[pad_positions], expected))

    def test_pretrain_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "toy.jsonl"
            rows = [{"text": "hello"}, {"text": "world"}, {"text": "mini"}]
            with path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            try:
                dataset = PretrainDataset(str(path), self.tokenizer, max_length=8)
                length = len(dataset)
                item = dataset[0]
            except NotImplementedError as exc:
                self.skipTest(str(exc))

            self.assertEqual(length, 3)
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            self.assertEqual(tuple(item[0].shape), (8,))
            self.assertEqual(tuple(item[1].shape), (8,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
