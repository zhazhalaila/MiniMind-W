import types

import pytest


class FakeTokenizer:
    def __init__(self):
        self.bos_token_id = 101
        self.eos_token_id = 102
        self.pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, max_length=None, truncation=False):
        token_ids = [ord(ch) % 50 + 10 for ch in text]
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        return types.SimpleNamespace(input_ids=token_ids)


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


@pytest.fixture
def sample_record():
    return {"text": "hello"}

