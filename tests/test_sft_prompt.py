import pytest

from scratch_sft.prompt import (
    build_sft_chat_prompt,
    build_sft_special_token_ids,
    format_sft_messages,
    generate_sft_labels,
    pad_sft_example,
    postprocess_sft_prompt,
)


def xfail_on_not_implemented(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.xfail(f"{fn.__name__} is not implemented yet.")


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


def test_format_sft_messages_returns_role_content_list():
    record = {
        "conversations": [
            {"role": "user", "content": "Introduce Hangzhou."},
            {"role": "assistant", "content": "Hangzhou is the capital of Zhejiang."},
        ]
    }
    actual = xfail_on_not_implemented(
        format_sft_messages,
        record=record,
        add_system_ratio=0.0,
    )
    assert isinstance(actual, list)
    assert all(isinstance(item, dict) for item in actual)
    assert all("role" in item and "content" in item for item in actual)


def test_build_sft_chat_prompt_returns_string():
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "Introduce Hangzhou."},
        {"role": "assistant", "content": "Hangzhou is the capital of Zhejiang."},
    ]
    actual = xfail_on_not_implemented(build_sft_chat_prompt, messages, tokenizer)
    assert isinstance(actual, str)


def test_postprocess_sft_prompt_returns_string():
    actual = xfail_on_not_implemented(
        postprocess_sft_prompt,
        prompt_text="<think>\n\n</think>\n\nassistant: hello",
        empty_think_ratio=0.2,
    )
    assert isinstance(actual, str)


def test_build_sft_special_token_ids_returns_two_lists():
    tokenizer = FakeTokenizer()
    bos_ids, eos_ids = xfail_on_not_implemented(build_sft_special_token_ids, tokenizer)
    assert isinstance(bos_ids, list)
    assert isinstance(eos_ids, list)


def test_generate_sft_labels_returns_parallel_list():
    labels = xfail_on_not_implemented(
        generate_sft_labels,
        input_ids=[11, 12, 13, 14, 15, 16],
        assistant_bos_ids=[11, 12],
        assistant_eos_ids=[15, 16],
        max_seq_len=6,
    )
    assert isinstance(labels, list)
    assert len(labels) == 6


def test_pad_sft_example_returns_fixed_length_lists():
    padded_input_ids, padded_labels = xfail_on_not_implemented(
        pad_sft_example,
        input_ids=[11, 12, 13],
        labels=[-100, -100, 13],
        pad_token_id=0,
        max_seq_len=8,
    )
    assert isinstance(padded_input_ids, list)
    assert isinstance(padded_labels, list)
    assert len(padded_input_ids) == 8
    assert len(padded_labels) == 8
