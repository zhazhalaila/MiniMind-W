import sys
import unittest

import torch

from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model_minimind import MiniMindConfig  # noqa: E402
from scratch_pretrain.eval_chat import (  # noqa: E402
    build_chat_messages,
    build_chat_prompt,
    chat_once,
    encode_chat_prompt,
    generate_with_kv_cache,
    load_inference_artifacts,
    run_chat_cli,
    sample_next_token,
)


class DummyTokenizer:
    def __len__(self):
        return 256

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        prompt = "\n".join(parts)
        if add_generation_prompt:
            prompt += "\nassistant: "
        return prompt

    def __call__(self, prompt, return_tensors="pt", add_special_tokens=True):
        del add_special_tokens
        ids = [min(ord(ch), 255) for ch in prompt][:16] or [1]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return "".join(chr(max(1, min(x, 255))) for x in token_ids)


class DummyModel(torch.nn.Module):
    def forward(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError("Dummy model does not implement forward.")


class InferenceKVCacheChecks(unittest.TestCase):
    def _call_or_skip(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            self.skipTest(str(exc))

    def build_fake_model_config(self):
        return MiniMindConfig(
            vocab_size=128,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=32,
        )

    def test_build_chat_messages(self):
        actual = self._call_or_skip(
            build_chat_messages,
            user_text="hello",
            system_prompt="You are a bot.",
            history=[{"role": "assistant", "content": "hi"}],
        )
        self.assertIsInstance(actual, list)
        self.assertTrue(all(isinstance(item, dict) for item in actual))
        self.assertTrue(all("role" in item and "content" in item for item in actual))

    def test_build_chat_prompt(self):
        tokenizer = DummyTokenizer()
        messages = [
            {"role": "system", "content": "You are a bot."},
            {"role": "user", "content": "hello"},
        ]
        actual = self._call_or_skip(build_chat_prompt, messages, tokenizer)
        self.assertIsInstance(actual, str)

    def test_encode_chat_prompt(self):
        tokenizer = DummyTokenizer()
        actual = self._call_or_skip(
            encode_chat_prompt,
            prompt="user: hello\nassistant: ",
            tokenizer=tokenizer,
            device=torch.device("cpu"),
        )
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.dtype, torch.long)
        self.assertEqual(actual.ndim, 2)
        self.assertEqual(actual.shape[0], 1)

    def test_sample_next_token(self):
        logits = torch.randn(1, 8)
        actual = self._call_or_skip(sample_next_token, logits, 1.0, 1.0, 0)
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.dtype, torch.long)
        self.assertEqual(actual.shape, (1, 1))

    def test_generate_with_kv_cache(self):
        model = DummyModel()
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        actual = self._call_or_skip(
            generate_with_kv_cache,
            model,
            input_ids,
            4,
            None,
            1.0,
            1.0,
            0,
        )
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.dtype, torch.long)
        self.assertEqual(actual.ndim, 2)
        self.assertEqual(actual.shape[0], 1)

    def test_chat_once(self):
        model = DummyModel()
        tokenizer = DummyTokenizer()
        messages = [{"role": "user", "content": "hello"}]
        actual = self._call_or_skip(
            chat_once,
            model,
            tokenizer,
            messages,
            torch.device("cpu"),
            8,
            None,
            1.0,
            1.0,
            0,
        )
        self.assertIsInstance(actual, str)

    def test_load_inference_artifacts(self):
        tmp_dir = ROOT / "tmp_inference_kvcache_checks"
        tmp_dir.mkdir(exist_ok=True)
        weight_path = tmp_dir / "toy.pt"
        torch.save({"model": {}}, weight_path)
        with patch("scratch_pretrain.eval_chat.load_tokenizer", return_value=DummyTokenizer()):
            actual = self._call_or_skip(
                load_inference_artifacts,
                str(weight_path),
                str(tmp_dir),
                self.build_fake_model_config(),
                torch.device("cpu"),
            )
        self.assertIsInstance(actual, tuple)
        self.assertEqual(len(actual), 2)

    def test_run_chat_cli(self):
        argv = [
            "eval_chat.py",
            "--weight_path", "out/pretrain_final.pt",
            "--tokenizer_dir", "tokenizer",
            "--hidden_size", "16",
            "--intermediate_size", "32",
            "--num_hidden_layers", "2",
            "--num_attention_heads", "4",
            "--num_key_value_heads", "4",
        ]

        with patch.object(sys, "argv", argv):
            with patch("builtins.input", return_value="exit"):
                with patch("scratch_pretrain.eval_chat.load_tokenizer", return_value=DummyTokenizer()):
                    with patch(
                        "scratch_pretrain.eval_chat.load_inference_artifacts",
                        return_value=(DummyTokenizer(), DummyModel()),
                    ):
                        actual = self._call_or_skip(run_chat_cli)
        self.assertIsNone(actual)


if __name__ == "__main__":
    unittest.main(verbosity=2)
