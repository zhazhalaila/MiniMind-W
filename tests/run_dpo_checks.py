import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch_dpo.config import build_dpo_data_config, build_dpo_train_config  # noqa: E402
from scratch_dpo.dataloader import build_dpo_dataloader, collate_dpo_batch  # noqa: E402
from scratch_dpo.dataset import (  # noqa: E402
    DPODataset,
    build_dpo_chat_prompt,
    build_dpo_pair_example,
    build_dpo_sequence_tensors,
    build_dpo_special_token_ids,
    generate_dpo_loss_mask,
    load_dpo_jsonl_records,
    postprocess_dpo_prompt,
)
from scratch_dpo.loss import dpo_loss, logits_to_log_probs, masked_sequence_log_probs, split_chosen_rejected  # noqa: E402
from scratch_dpo.train_dpo import build_dpo_parser, build_model_config_from_args  # noqa: E402
from scratch_dpo.train_loop import concat_chosen_rejected_batch, move_dpo_batch_to_device  # noqa: E402


class FakeTokenizer:
    bos_token = "<bos>"
    eos_token = "<eos>"
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None):
        del tokenize
        del add_generation_prompt
        del tools
        return "\n".join(f"{item['role']}: {item['content']}" for item in messages)

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None, padding=None):
        del add_special_tokens
        token_ids = [ord(ch) % 50 + 10 for ch in text]
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        if padding == "max_length" and max_length is not None:
            token_ids = token_ids + [self.pad_token_id] * max(0, max_length - len(token_ids))
        return {"input_ids": token_ids}


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


class DPOChecks(unittest.TestCase):
    def _call_or_skip(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            self.skipTest(str(exc))

    def test_build_dpo_configs(self):
        data_config = self._call_or_skip(build_dpo_data_config, "tokenizer", "data/dpo.jsonl", 1024, 0.2)
        train_config = self._call_or_skip(
            build_dpo_train_config,
            "logs",
            "checkpoints",
            "out",
            "dpo",
            "out/full_sft_dense/full_sft_768_final.pt",
            "none",
            1,
            4,
            4e-8,
            0.0,
            "cpu",
            "float32",
            0,
            1,
            1.0,
            1,
            100,
            0,
            0.1,
            0.15,
        )
        self.assertEqual(data_config.max_seq_len, 1024)
        self.assertEqual(train_config.beta, 0.15)

    def test_load_dpo_jsonl_records(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "toy_dpo.jsonl"
            rows = [
                {
                    "chosen": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
                    "rejected": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "bad"}],
                }
            ]
            with path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            actual = self._call_or_skip(load_dpo_jsonl_records, str(path))
            self.assertEqual(len(actual), 1)

    def test_dpo_dataset_sample_keys(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "toy_dpo.jsonl"
            row = {
                "chosen": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
                "rejected": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "bad"}],
            }
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            dataset = self._call_or_skip(DPODataset, str(path), FakeTokenizer(), 16, 0.2)
            sample = dataset[0]
            self.assertEqual(
                set(sample.keys()),
                {"x_chosen", "y_chosen", "mask_chosen", "x_rejected", "y_rejected", "mask_rejected"},
            )

    def test_prompt_and_mask_helpers(self):
        tokenizer = FakeTokenizer()
        bos_ids, eos_ids = self._call_or_skip(build_dpo_special_token_ids, tokenizer)
        prompt = self._call_or_skip(
            build_dpo_chat_prompt,
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
            tokenizer,
        )
        processed = self._call_or_skip(postprocess_dpo_prompt, prompt, 0.2)
        input_ids = tokenizer(processed)["input_ids"][:16]
        mask = self._call_or_skip(generate_dpo_loss_mask, input_ids, bos_ids, eos_ids, 16)
        self.assertIsInstance(mask, list)

    def test_build_dpo_sequence_tensors(self):
        x, y, mask = self._call_or_skip(build_dpo_sequence_tensors, [1, 2, 3, 0], [0, 1, 1, 0], 0, 6)
        self.assertEqual(x.shape, (5,))
        self.assertEqual(y.shape, (5,))
        self.assertEqual(mask.shape, (5,))

    def test_build_dpo_pair_example(self):
        tokenizer = FakeTokenizer()
        bos_ids, eos_ids = self._call_or_skip(build_dpo_special_token_ids, tokenizer)
        record = {
            "chosen": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
            "rejected": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "bad"}],
        }
        actual = self._call_or_skip(build_dpo_pair_example, record, tokenizer, bos_ids, eos_ids, 16, 0.2)
        self.assertEqual(
            set(actual.keys()),
            {"x_chosen", "y_chosen", "mask_chosen", "x_rejected", "y_rejected", "mask_rejected"},
        )

    def test_collate_and_concat_dpo_batch(self):
        sample = ToyDPODataset()[0]
        batch = self._call_or_skip(collate_dpo_batch, [sample, sample])
        self.assertEqual(batch["x_chosen"].shape, (2, 3))
        x, y, mask = self._call_or_skip(concat_chosen_rejected_batch, batch)
        self.assertEqual(x.shape, (4, 3))
        self.assertEqual(y.shape, (4, 3))
        self.assertEqual(mask.shape, (4, 3))

    def test_build_dpo_dataloader(self):
        dataloader = self._call_or_skip(build_dpo_dataloader, ToyDPODataset(), 2, False)
        batch = next(iter(dataloader))
        self.assertEqual(batch["x_chosen"].shape, (2, 3))

    def test_dpo_loss_helpers(self):
        logits = torch.randn(4, 3, 7)
        labels = torch.randint(0, 7, (4, 3))
        token_log_probs = self._call_or_skip(logits_to_log_probs, logits, labels)
        self.assertEqual(token_log_probs.shape, (4, 3))
        seq_log_probs = self._call_or_skip(masked_sequence_log_probs, token_log_probs, torch.ones(4, 3))
        self.assertEqual(seq_log_probs.shape, (4,))
        chosen, rejected = self._call_or_skip(split_chosen_rejected, seq_log_probs)
        self.assertEqual(chosen.shape, rejected.shape)
        loss = self._call_or_skip(dpo_loss, token_log_probs, token_log_probs, torch.ones(4, 3), 0.15)
        self.assertEqual(loss.shape, ())

    def test_train_parser_and_model_config(self):
        parser = self._call_or_skip(build_dpo_parser)
        self.assertIsInstance(parser, argparse.ArgumentParser)
        args = parser.parse_args([])
        config = self._call_or_skip(build_model_config_from_args, args)
        self.assertTrue(hasattr(config, "hidden_size"))

    def test_move_batch_to_device(self):
        sample = ToyDPODataset()[0]
        batch = {key: value.unsqueeze(0) for key, value in sample.items()}
        moved = self._call_or_skip(move_dpo_batch_to_device, batch, "cpu")
        self.assertEqual(moved["x_chosen"].device.type, "cpu")


if __name__ == "__main__":
    unittest.main(verbosity=2)

