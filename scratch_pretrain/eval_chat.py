from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import argparse
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model_minimind import MiniMindConfig
from scratch_pretrain.entry import build_model, load_checkpoint_file
from scratch_pretrain.tokenizer_utils import load_tokenizer

def load_inference_artifacts(
    weight_path: str,
    tokenizer_dir: str,
    model_config: MiniMindConfig,
    device: Union[str, torch.device],
) -> Tuple[Any, nn.Module]:
    """
    Load tokenizer + model weights for local chat inference.

    Input:
        weight_path: str
        tokenizer_dir: str
        model_config: MiniMindConfig
        device: str | torch.device

    Output:
        tokenizer: Any
        model: torch.nn.Module
    """

    tokenizer = load_tokenizer(tokenizer_dir)

    model = build_model(
        model_config=model_config,
        device=device,
    )

    payload = load_checkpoint_file(
        checkpoint_path=weight_path,
        device=device,
    )

    if isinstance(payload, dict) and "model" in payload:
        state_dict = payload["model"]
    else:
        state_dict = payload

    if state_dict:
        model.load_state_dict(state_dict, strict=True)
    model.eval()

    return tokenizer, model


def build_chat_messages(
    user_text: str,
    system_prompt: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """
    Build chat messages for one inference round.

    Input:
        user_text: str
        system_prompt: str | None
        history: list[dict[str, str]] | None

    Output:
        messages: list[dict[str, str]]
    """

    messages: List[Dict[str, str]] = []

    if system_prompt is not None and system_prompt.strip():
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

    if history is not None:
        for item in history:
            messages.append(
                {
                    "role": item["role"],
                    "content": item["content"],
                }
            )

    messages.append(
        {
            "role": "user",
            "content": user_text,
        }
    )

    return messages


def build_chat_prompt(
    messages: List[Dict[str, str]],
    tokenizer: Any,
    open_thinking: bool = False,
) -> str:
    """
    Convert chat messages into one prompt string.

    Input:
        messages: list[dict[str, str]]
        tokenizer: Any

    Output:
        prompt: str
    """

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                open_thinking=open_thinking,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt
    
    parts = []
    for item in messages:
        role = item["role"]
        content = item["content"]
        parts.append(f"{role}: {content}")

    parts.append("assistant: ")
    prompt = "\n".join(parts)
    return prompt
    


def encode_chat_prompt(
    prompt: str,
    tokenizer: Any,
    device: Union[str, torch.device],
) -> torch.Tensor:
    """
    Encode one chat prompt into model input ids.

    Input:
        prompt: str
        tokenizer: Any
        device: str | torch.device

    Output:
        input_ids: torch.Tensor
            Shape: (1, L)
    """

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )

    input_ids = encoded["input_ids"].to(device)
    return input_ids


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    """
    Sample one next token from the last-position logits.

    Input:
        logits: torch.Tensor
            Shape: (1, V)
        temperature: float
        top_p: float
        top_k: int

    Output:
        next_token: torch.Tensor
            Shape: (1, 1)
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    
    logits = logits / temperature

    if top_k > 0:
        topk_values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
        kth_value = topk_values[:, -1].unsqueeze(-1) # shape (1, 1)
        logits = torch.where(
            logits < kth_value,
            torch.full_like(logits, float("-inf")),
            logits,
        ) # just consider top-k

    probs = torch.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_indices.gather(-1, sampled_idx)
        return next_token
    
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def generate_with_kv_cache(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    """
    Generate tokens autoregressively with KV cache.

    Input:
        model: torch.nn.Module
        input_ids: torch.Tensor
            Shape: (1, L)
        max_new_tokens: int
        eos_token_id: int | None
        temperature: float
        top_p: float
        top_k: int

    Output:
        output_ids: torch.Tensor
            Shape: (1, L_new)
    """

    model.eval()

    generated_ids = input_ids
    current_input_ids = input_ids
    past_key_values = None

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # the first iteration is prefill, the input shape is (1, L)
            # beside first iteration, other is decode, the input shape is (1, 1) with past_key_values
            outputs = model(
                input_ids=current_input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )

            # (1, L, V)
            logits = outputs.logits
            past_key_values = outputs.past_key_values

            next_token_logits = logits[:, -1, :]

            next_token = sample_next_token(
                logits=next_token_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

            current_input_ids = next_token

    return generated_ids

def chat_once(
    model: nn.Module,
    tokenizer: Any,
    messages: List[Dict[str, str]],
    device: Union[str, torch.device],
    max_new_tokens: int = 8192,
    eos_token_id: Optional[int] = None,
    temperature: float = 0.85,
    top_p: float = 0.95,
    top_k: int = 0,
    open_thinking: bool = False,
) -> str:
    """
    Generate one assistant response from chat messages.

    Input:
        model: torch.nn.Module
        tokenizer: Any
        messages: list[dict[str, str]]
        device: str | torch.device
        max_new_tokens: int
        eos_token_id: int | None
        temperature: float
        top_p: float
        top_k: int

    Output:
        response: str
    """

    prompt = build_chat_prompt(messages, tokenizer, open_thinking=open_thinking)
    input_ids = encode_chat_prompt(prompt, tokenizer, device)

    output_ids = generate_with_kv_cache(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    new_token_ids = output_ids[:, input_ids.size(1):]

    response = tokenizer.decode(
        new_token_ids[0],
        skip_special_tokens=True,
    )

    return response.strip()

def run_chat_cli() -> None:
    """
    Run one minimal local chat CLI.
    """

    parser = argparse.ArgumentParser("eval_chat")

    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--tokenizer_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--eos_token_id", type=int, default=None)
    parser.add_argument("--open_thinking", type=int, default=0, choices=[0, 1])
    parser.add_argument("--historys", type=int, default=0)
    parser.add_argument("--show_speed", type=int, default=1, choices=[0, 1])

    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, required=True)
    parser.add_argument("--intermediate_size", type=int, required=True)
    parser.add_argument("--num_hidden_layers", type=int, required=True)
    parser.add_argument("--num_attention_heads", type=int, required=True)
    parser.add_argument("--num_key_value_heads", type=int, required=True)
    parser.add_argument("--max_position_embeddings", type=int, default=32768)
    parser.add_argument("--rope_theta", type=float, default=1000000.0)
    parser.add_argument("--rms_norm_eps", type=float, default=1e-6)

    parser.add_argument("--use_moe", action="store_true")
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--num_experts_per_tok", type=int, default=1)
    parser.add_argument("--moe_intermediate_size", type=int, default=None)
    parser.add_argument("--router_aux_loss_coef", type=float, default=5e-4)

    args = parser.parse_args()

    device = torch.device(args.device)

    tokenizer = load_tokenizer(args.tokenizer_dir)
    vocab_size = args.vocab_size if args.vocab_size is not None else len(tokenizer)

    moe_intermediate_size = (
        args.moe_intermediate_size
        if args.moe_intermediate_size is not None
        else args.intermediate_size
    )

    model_config = MiniMindConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        max_position_embeddings=args.max_position_embeddings,
        rope_theta=args.rope_theta,
        rms_norm_eps=args.rms_norm_eps,
        use_moe=args.use_moe,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        router_aux_loss_coef=args.router_aux_loss_coef,
    )

    tokenizer, model = load_inference_artifacts(
        weight_path=args.weight_path,
        tokenizer_dir=args.tokenizer_dir,
        model_config=model_config,
        device=device,
    )

    eos_token_id = args.eos_token_id
    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

    history: list[dict[str, str]] = []

    print("MiniMind chat cli")
    print("type `/exit` to quit, `/clear` to clear history")

    while True:
        try:
            user_text = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue

        if user_text in {"/exit", "exit", "quit"}:
            break

        if user_text == "/clear":
            history.clear()
            print("history cleared")
            continue

        current_history = history[-args.historys:] if args.historys else []
        messages = build_chat_messages(
            user_text=user_text,
            system_prompt=args.system_prompt,
            history=current_history,
        )

        start_time = time.time()
        response = chat_once(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            device=device,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_token_id,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            open_thinking=bool(args.open_thinking),
        )

        print(f"assistant> {response}")
        if args.show_speed:
            elapsed = max(time.time() - start_time, 1e-6)
            encoded_response = tokenizer(response, add_special_tokens=False)
            if isinstance(encoded_response, dict):
                response_token_ids = encoded_response["input_ids"]
            else:
                response_token_ids = encoded_response.input_ids
            generated_tokens = len(response_token_ids)
            print(f"[Speed]: {generated_tokens / elapsed:.2f} tokens/s")

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": response})


def main() -> None:
    run_chat_cli()


if __name__ == "__main__":
    main()
