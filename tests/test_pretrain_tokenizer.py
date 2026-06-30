from unittest.mock import patch

import pytest

from scratch_pretrain.tokenizer_utils import load_tokenizer


def xfail_on_not_implemented(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.xfail(f"{fn.__name__} is not implemented yet.")


def test_load_tokenizer_uses_auto_tokenizer():
    fake_tokenizer = object()
    fake_auto_tokenizer = type("FakeAutoTokenizer", (), {"from_pretrained": staticmethod(lambda path: fake_tokenizer)})

    with patch("scratch_pretrain.tokenizer_utils.AutoTokenizer", fake_auto_tokenizer):
        actual = xfail_on_not_implemented(load_tokenizer, "tokenizer")

    assert actual is fake_tokenizer
