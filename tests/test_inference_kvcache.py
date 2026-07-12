import sys

import pytest
import torch

from model.model_minimind import MiniMindConfig
from scratch_pretrain.eval_chat import (
    build_chat_messages,
    build_chat_prompt,
    chat_once,
    encode_chat_prompt,
    generate_with_kv_cache,
    load_inference_artifacts,
    run_chat_cli,
    sample_next_token,
)


def xfail_on_not_implemented(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.xfail(f"{fn.__name__} is not implemented yet.")


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


def build_fake_model_config() -> MiniMindConfig:
    return MiniMindConfig(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
    )


def test_build_chat_messages_returns_role_content_list():
    actual = xfail_on_not_implemented(
        build_chat_messages,
        user_text="hello",
        system_prompt="You are a bot.",
        history=[{"role": "assistant", "content": "hi"}],
    )
    assert isinstance(actual, list)
    assert all(isinstance(item, dict) for item in actual)
    assert all("role" in item and "content" in item for item in actual)


def test_build_chat_prompt_returns_string():
    tokenizer = DummyTokenizer()
    messages = [
        {"role": "system", "content": "You are a bot."},
        {"role": "user", "content": "hello"},
    ]
    actual = xfail_on_not_implemented(build_chat_prompt, messages, tokenizer)
    assert isinstance(actual, str)


def test_encode_chat_prompt_returns_rank2_long_tensor():
    tokenizer = DummyTokenizer()
    actual = xfail_on_not_implemented(
        encode_chat_prompt,
        prompt="user: hello\nassistant: ",
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )
    assert isinstance(actual, torch.Tensor)
    assert actual.dtype == torch.long
    assert actual.ndim == 2
    assert actual.shape[0] == 1


def test_sample_next_token_returns_rank2_long_tensor():
    logits = torch.randn(1, 8)
    actual = xfail_on_not_implemented(sample_next_token, logits, 1.0, 1.0, 0)
    assert isinstance(actual, torch.Tensor)
    assert actual.dtype == torch.long
    assert actual.shape == (1, 1)


def test_generate_with_kv_cache_returns_rank2_long_tensor():
    model = DummyModel()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    actual = xfail_on_not_implemented(
        generate_with_kv_cache,
        model,
        input_ids,
        4,
        None,
        1.0,
        1.0,
        0,
    )
    assert isinstance(actual, torch.Tensor)
    assert actual.dtype == torch.long
    assert actual.ndim == 2
    assert actual.shape[0] == 1


def test_chat_once_returns_string():
    model = DummyModel()
    tokenizer = DummyTokenizer()
    messages = [{"role": "user", "content": "hello"}]
    actual = xfail_on_not_implemented(
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
    assert isinstance(actual, str)


def test_load_inference_artifacts_returns_tokenizer_and_model(tmp_path, monkeypatch):
    weight_path = tmp_path / "toy.pt"
    torch.save({"model": {}}, weight_path)
    monkeypatch.setattr(
        "scratch_pretrain.eval_chat.load_tokenizer",
        lambda tokenizer_dir: DummyTokenizer(),
    )
    actual = xfail_on_not_implemented(
        load_inference_artifacts,
        str(weight_path),
        str(tmp_path),
        build_fake_model_config(),
        torch.device("cpu"),
    )
    assert isinstance(actual, tuple)
    assert len(actual) == 2


def test_run_chat_cli_runs_without_return_value(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_chat.py",
            "--weight_path", "out/pretrain_final.pt",
            "--tokenizer_dir", "tokenizer",
            "--hidden_size", "16",
            "--intermediate_size", "32",
            "--num_hidden_layers", "2",
            "--num_attention_heads", "4",
            "--num_key_value_heads", "4",
        ],
    )
    monkeypatch.setattr("builtins.input", lambda _: "exit")
    monkeypatch.setattr(
        "scratch_pretrain.eval_chat.load_tokenizer",
        lambda tokenizer_dir: DummyTokenizer(),
    )
    monkeypatch.setattr(
        "scratch_pretrain.eval_chat.load_inference_artifacts",
        lambda **kwargs: (DummyTokenizer(), DummyModel()),
    )
    actual = xfail_on_not_implemented(run_chat_cli)
    assert actual is None
