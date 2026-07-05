from __future__ import annotations

from typing import Any

try:
    from transformers import AutoTokenizer
except ModuleNotFoundError:  # pragma: no cover
    AutoTokenizer = None


def load_tokenizer(tokenizer_dir: str) -> Any:
    """
    Load a tokenizer from a local directory.

    Input:
        tokenizer_dir: str

    Output:
        tokenizer: Any
    """
    if AutoTokenizer is None:
        raise ModuleNotFoundError("transformers is not installed")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    return tokenizer
