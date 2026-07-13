from __future__ import annotations

import random, json
from typing import Any, Dict, List


def format_sft_messages(
    record: Dict[str, Any],
    add_system_ratio: float = 0.2,
) -> List[Dict[str, Any]]:
    """
    Normalize one raw SFT record into MiniMind-style chat messages.

    Input:
        record: dict[str, Any]
            Expected:
                record["conversations"]: list[dict[str, Any]]
        add_system_ratio: float
            Probability used when injecting one synthetic `system` turn
            for records that do not already start with `system`.

    Output:
        messages: list[dict[str, Any]]
            Each message should preserve at least:
                - role: str
                - content: str
            Optional fields may also be present:
                - reasoning_content
                - tools
                - tool_calls
    """

    if not isinstance(record, dict):
        raise ValueError("record must be a dict.")

    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("record['conversations'] must be a non-empty list.")

    if not isinstance(add_system_ratio, (int, float)) or not 0.0 <= float(add_system_ratio) <= 1.0:
        raise ValueError("add_system_ratio must be in [0, 1].")

    messages = []
    for message in conversations:
        if not isinstance(message, dict):
            raise ValueError("each conversation message must be a dict.")
        if "role" not in message or "content" not in message:
            raise ValueError("each message must contain 'role' and 'content'.")
        messages.append(dict(message))

    # tool
    if any(msg.get("tools") for msg in messages):
        return messages

    # add random system prompt
    if messages[0].get("role") != "system":
        system_prompts = [
            "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
            "你是minimind，一个小巧但有用的语言模型。",
            "你是一个专业的AI助手，请提供有价值的回答。",
            "你是minimind，请尽力帮助用户解决问题。",
            "你是一个可靠的AI，请给出准确的回答。",
            "You are a helpful AI assistant.",
            "You are minimind, a lightweight intelligent assistant.",
            "You are a friendly chatbot. Please answer the user's questions carefully.",
            "You are a knowledgeable AI. Try your best to provide accurate information.",
            "You are minimind, a small but useful language model.",
        ]

        if random.random() < float(add_system_ratio):
            messages = [
                {
                    "role": "system",
                    "content": random.choice(system_prompts),
                }
            ] + messages

    return messages

def build_sft_chat_prompt(
    messages: List[Dict[str, Any]],
    tokenizer: Any,
) -> str:
    """
    Build one full MiniMind-style chat prompt with `apply_chat_template`.

    Input:
        messages: list[dict[str, Any]]
            Multi-turn conversation messages in MiniMind SFT format.
        tokenizer: Any

    Output:
        prompt_text: str

    Notes:
        Expected template call shape:
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                tools=tools,
            )
    """

    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list.")
    
    normalized_messages = []
    tools = None

    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be a dict.")
        
        item = dict(message)

        if item.get("role") == "system" and item.get("tools"):
            tools = json.loads(item["tools"]) if isinstance(item["tools"], str) else item["tools"]

        if item.get("tool_calls") and isinstance(item["tool_calls"], str):
            item["tool_calls"] = json.loads(item["tool_calls"])

        normalized_messages.append(item)

    prompt_text = tokenizer.apply_chat_template(
        normalized_messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools,
    )

    return prompt_text


def postprocess_sft_prompt(
    prompt_text: str,
    empty_think_ratio: float = 0.2,
) -> str:
    """
    Apply MiniMind-style prompt postprocessing before tokenization.

    Input:
        prompt_text: str
        empty_think_ratio: float
            Probability of keeping one empty `<think>` block.

    Output:
        processed_prompt_text: str
    """

    if not isinstance(prompt_text, str):
        raise ValueError("prompt_text must be a string.")

    if not isinstance(empty_think_ratio, (int, float)) or not 0.0 <= float(empty_think_ratio) <= 1.0:
        raise ValueError("empty_think_ratio must be in [0, 1].")

    empty_think = "<think>\n\n</think>\n\n"

    if empty_think in prompt_text and random.random() > float(empty_think_ratio):
        prompt_text = prompt_text.replace(empty_think, "")

    return prompt_text


def build_sft_special_token_ids(
    tokenizer: Any,
) -> tuple[list[int], list[int]]:
    """
    Build assistant start/end marker token ids used for SFT label masking.

    Input:
        tokenizer: Any
            Expected to expose:
                - bos_token
                - eos_token

    Output:
        assistant_bos_ids: list[int]
            Marker for one assistant span start.
        assistant_eos_ids: list[int]
            Marker for one assistant span end.

    Notes:
        MiniMind-style expected markers:
            - f'{tokenizer.bos_token}assistant\\n'
            - f'{tokenizer.eos_token}\\n'
    """

    if not hasattr(tokenizer, "bos_token") or tokenizer.bos_token is None:
        raise ValueError("tokenizer must have bos_token.")

    if not hasattr(tokenizer, "eos_token") or tokenizer.eos_token is None:
        raise ValueError("tokenizer must have eos_token.")
    
    assistant_bos_ids = tokenizer(
        f"{tokenizer.bos_token}assistant\n",
        add_special_tokens=False,
    ).input_ids

    assistant_eos_ids = tokenizer(
        f"{tokenizer.eos_token}assistant\n",
        add_special_tokens=False,
    ).input_ids

    return assistant_bos_ids, assistant_eos_ids


def generate_sft_labels(
    input_ids: list[int],
    assistant_bos_ids: list[int],
    assistant_eos_ids: list[int],
    max_seq_len: int,
) -> list[int]:
    """
    Generate MiniMind-style SFT labels by supervising only assistant spans.

    Input:
        input_ids: list[int]
            Shape: (L,)
        assistant_bos_ids: list[int]
        assistant_eos_ids: list[int]
        max_seq_len: int

    Output:
        labels: list[int]
            Shape: (L,)

    Notes:
        Expected masking rule:
            - non-assistant tokens -> -100
            - assistant response tokens -> original token ids

    Examples:
        input_ids = [
            10, 11,          # user prompt
            1, 20, 21,       # assistant_bos_ids
            31, 32, 33,      # assistant response
            2, 99,           # assistant_eos_ids
            40, 41           # user prompt
        ]

        labels = [
            -100, -100,
            -100, -100, -100,
            31, 32, 33,
            2, 99, # not mask end token, since we need to learn when to stop
            -100, -100
        ]

    """

    if not isinstance(input_ids, list):
        raise ValueError("input_ids must be a list.")

    if not all(isinstance(token_id, int) for token_id in input_ids):
        raise ValueError("input_ids must be a list[int].")

    if not isinstance(assistant_bos_ids, list) or not assistant_bos_ids:
        raise ValueError("assistant_bos_ids must be a non-empty list.")

    if not all(isinstance(token_id, int) for token_id in assistant_bos_ids):
        raise ValueError("assistant_bos_ids must be a list[int].")

    if not isinstance(assistant_eos_ids, list) or not assistant_eos_ids:
        raise ValueError("assistant_eos_ids must be a non-empty list.")

    if not all(isinstance(token_id, int) for token_id in assistant_eos_ids):
        raise ValueError("assistant_eos_ids must be a list[int].")

    if not isinstance(max_seq_len, int) or max_seq_len <= 0:
        raise ValueError("max_seq_len must be a positive integer.")
    
    # init all labels to -100, we just crossloss assistant response not user prompt
    labels = [-100] * len(input_ids)

    i = 0
    while i < len(input_ids):
        if input_ids[i : i + len(assistant_bos_ids)] == assistant_bos_ids:
            start = i + len(assistant_bos_ids)
            end = start

            # find assistant response
            while end < len(input_ids):
                if input_ids[end : end + len(assistant_eos_ids)] == assistant_eos_ids:
                    break
                end += 1

            label_end = min(end + len(assistant_eos_ids), max_seq_len, len(input_ids))

            for j in range(start, label_end):
                labels[j] = input_ids[j]

            if end < len(input_ids):
                i = end + len(assistant_eos_ids)
            else:
                i = len(input_ids)
        else:
            i += 1

    return labels

def pad_sft_example(
    input_ids: list[int],
    labels: list[int],
    pad_token_id: int,
    max_seq_len: int,
) -> tuple[list[int], list[int]]:
    """
    Pad or truncate one SFT example to fixed length.

    Input:
        input_ids: list[int]
            Shape: (L,)
        labels: list[int]
            Shape: (L,)
        pad_token_id: int
        max_seq_len: int

    Output:
        padded_input_ids: list[int]
            Shape: (max_seq_len,)
        padded_labels: list[int]
            Shape: (max_seq_len,)
    """

    if not isinstance(input_ids, list):
        raise ValueError("input_ids must be a list.")

    if not isinstance(labels, list):
        raise ValueError("labels must be a list.")

    if len(input_ids) != len(labels):
        raise ValueError("input_ids and labels must have the same length.")

    if not all(isinstance(token_id, int) for token_id in input_ids):
        raise ValueError("input_ids must be a list[int].")

    if not all(isinstance(label_id, int) for label_id in labels):
        raise ValueError("labels must be a list[int].")

    if not isinstance(pad_token_id, int):
        raise ValueError("pad_token_id must be an int.")

    if not isinstance(max_seq_len, int) or max_seq_len <= 0:
        raise ValueError("max_seq_len must be a positive integer.")

    input_ids = input_ids[:max_seq_len]
    labels = labels[:max_seq_len]

    pad_len = max_seq_len - len(input_ids)

    if pad_len > 0:
        input_ids = input_ids + [pad_token_id] * pad_len
        labels = labels + [-100] * pad_len

    return input_ids, labels
