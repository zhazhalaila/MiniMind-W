# KV Cache 对话推理

把对话推理链路搭起来，支持 `KV Cache`。

## 主线

```text
messages
-> chat prompt
-> tokenizer
-> input_ids
-> first forward(use_cache=True)
-> logits[:, -1, :]
-> sample next token
-> next forward(last_token + past_key_values)
-> append token
-> decode
```

## 输入与输出

### 输入

- 对话历史：
  - `messages: list[dict[str, str]]`
- 编码后输入：
  - `input_ids: torch.Tensor`
    - shape: `(1, L)`
- 缓存：
  - `past_key_values`
    - 每层一个 `tuple[k, v]`

### 输出

- 生成后的 token：
  - `output_ids: torch.Tensor`
    - shape: `(1, L_new)`
- 单轮回复：
  - `response: str`
- 新缓存：
  - `past_key_values`

## 需要完成的函数

### `model/model_minimind.py`

#### `Attention.forward(x, attention_mask=None, past_key_value=None, use_cache=False, position_offset=0)`

作用：

- 单层 attention 支持增量解码和 `KV Cache`。

输入：

- `x: torch.Tensor`
  - shape: `(B, L, D)`
- `attention_mask: torch.Tensor | None`
- `past_key_value: tuple[torch.Tensor, torch.Tensor] | None`
  - `past_k.shape == (B, Hkv, T, Hd)`
  - `past_v.shape == (B, Hkv, T, Hd)`
- `use_cache: bool`
- `position_offset: int`

输出：

- `attn_output: torch.Tensor`
  - shape: `(B, L, D)`
- `new_past_key_value: tuple[torch.Tensor, torch.Tensor] | None`
  - `k.shape == (B, Hkv, T + L, Hd)`
  - `v.shape == (B, Hkv, T + L, Hd)`

#### `DecoderLayer.forward(x, attention_mask=None, past_key_value=None, use_cache=False, position_offset=0)`

作用：

- 在 decoder block 内把 attention 输出和当前层 cache 一起往上传。

输入：

- `x: torch.Tensor`
  - shape: `(B, L, D)`
- `past_key_value: tuple[torch.Tensor, torch.Tensor] | None`
- `use_cache: bool`

输出：

- `hidden_states: torch.Tensor`
  - shape: `(B, L, D)`
- `new_past_key_value: tuple[torch.Tensor, torch.Tensor] | None`

#### `MiniMindModel.forward(input_ids, attention_mask=None, past_key_values=None, use_cache=False)`

作用：

- 把所有层的 `past_key_values` 一起接进来，并返回新的 `past_key_values`。

输入：

- `input_ids: torch.Tensor`
  - shape: `(B, L)`
- `past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None`

输出：

- `last_hidden_state: torch.Tensor`
  - shape: `(B, L, D)`
- `past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None`

#### `MiniMindForCausalLM.forward(input_ids, labels=None, attention_mask=None, past_key_values=None, use_cache=False)`

作用：

- 让 `Causal LM` 前向在推理时也能返回 `past_key_values`。

输入：

- `input_ids: torch.Tensor`
  - shape: `(B, L)`
- `labels: torch.Tensor | None`
  - shape: `(B, L)`
- `past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None`
- `use_cache: bool`

输出：

- `logits: torch.Tensor`
  - shape: `(B, L, V)`
- `loss: torch.Tensor | None`
  - shape: `()`
- `past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None`

### `scratch_pretrain/eval_chat.py`

#### `load_inference_artifacts(weight_path, tokenizer_dir, model_config, device) -> tuple[tokenizer, nn.Module]`

作用：

- 加载 tokenizer、模型配置和训练后的权重，构造推理模型。

输入：

- `weight_path: str`
- `tokenizer_dir: str`
- `model_config: MiniMindConfig`
- `device: str | torch.device`

输出：

- `tokenizer`
- `model: nn.Module`

#### `build_chat_messages(user_text, system_prompt=None, history=None) -> list[dict[str, str]]`

作用：

- 组装单轮或多轮对话的 `messages`。

输入：

- `user_text: str`
- `system_prompt: str | None`
- `history: list[dict[str, str]] | None`

输出：

- `messages: list[dict[str, str]]`

#### `build_chat_prompt(messages, tokenizer) -> str`

作用：

- 用 chat template 把 `messages` 拼成模型真正要看的 prompt。

输入：

- `messages: list[dict[str, str]]`
- `tokenizer`

输出：

- `prompt: str`

#### `encode_chat_prompt(prompt, tokenizer, device) -> torch.Tensor`

作用：

- 把 prompt 编码成推理输入张量。

输入：

- `prompt: str`
- `tokenizer`
- `device: str | torch.device`

输出：

- `input_ids: torch.Tensor`
  - shape: `(1, L)`

#### `sample_next_token(logits, temperature=1.0, top_p=1.0, top_k=0) -> torch.Tensor`

作用：

- 从最后一个位置的 logits 中采样下一个 token。

输入：

- `logits: torch.Tensor`
  - shape: `(1, V)`
- `temperature: float`
- `top_p: float`
- `top_k: int`

输出：

- `next_token: torch.Tensor`
  - shape: `(1, 1)`

#### `generate_with_kv_cache(model, input_ids, max_new_tokens, eos_token_id=None, temperature=1.0, top_p=1.0, top_k=0) -> torch.Tensor`

作用：

- 用 `KV Cache` 做自回归生成。

输入：

- `model: nn.Module`
- `input_ids: torch.Tensor`
  - shape: `(1, L)`
- `max_new_tokens: int`
- `eos_token_id: int | None`

输出：

- `output_ids: torch.Tensor`
  - shape: `(1, L_new)`

#### `chat_once(model, tokenizer, messages, device, max_new_tokens=128, eos_token_id=None, temperature=1.0, top_p=1.0, top_k=0) -> str`

作用：

- 完成一次从 `messages` 到 assistant 回复文本的生成。

输入：

- `messages: list[dict[str, str]]`
- 其余为模型和采样参数

输出：

- `response: str`

#### `run_chat_cli() -> None`

作用：

- 提供一个最小命令行对话入口。

输入：

- 无

输出：

- 无

## 本地测试命令

```bash
python3 tests/run_inference_kvcache_checks.py
```

## 推理命令

推理时优先加载 `out/` 下面导出的最终权重，不要直接拿 `checkpoints/` 里的训练态断点做日常对话。

- Dense 版优先加载：`out/pretrain_dense/pretrain_dense_final.pt`
- MoE 版优先加载：`out/pretrain_moe/pretrain_768_moe_final.pt`

### Dense 正式推理

```bash
python3 scratch_pretrain/eval_chat.py \
  --weight_path out/pretrain_dense/pretrain_dense_final.pt \
  --tokenizer_dir tokenizer \
  --device cuda:0 \
  --max_new_tokens 512 \
  --temperature 0.85 \
  --top_p 0.90 \
  --top_k 50 \
  --hidden_size 768 \
  --intermediate_size 2048 \
  --num_hidden_layers 8 \
  --num_attention_heads 8 \
  --num_key_value_heads 2
```

如果希望回复更稳定一些，可以把采样参数收紧为：

```text
temperature = 0.7
top_p = 0.8
top_k = 40
```

### MoE 正式推理

```bash
python3 scratch_pretrain/eval_chat.py \
  --weight_path out/pretrain_moe/pretrain_768_moe_final.pt \
  --tokenizer_dir tokenizer \
  --device cuda:0 \
  --max_new_tokens 512 \
  --temperature 0.85 \
  --top_p 0.90 \
  --top_k 50 \
  --hidden_size 768 \
  --intermediate_size 2432 \
  --num_hidden_layers 8 \
  --num_attention_heads 8 \
  --num_key_value_heads 4 \
  --max_position_embeddings 32768 \
  --rope_theta 1000000 \
  --rms_norm_eps 1e-6 \
  --use_moe \
  --num_experts 4 \
  --num_experts_per_tok 1 \
  --moe_intermediate_size 2432 \
  --router_aux_loss_coef 5e-4
```

### 多轮对话入口

上面两条命令启动后，会进入一个最小 CLI：

```text
MiniMind chat cli
type `/exit` to quit, `/clear` to clear history
```

可用命令：

- 输入普通文本：继续多轮对话
- 输入 `/clear`：清空历史
- 输入 `/exit`：退出