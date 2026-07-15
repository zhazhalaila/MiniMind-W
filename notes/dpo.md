# DPO 阶段

用偏好数据继续训练 SFT 模型，让模型更偏向 `chosen`，更远离 `rejected`。

与Pretrain的区别：

1. 输入的数据格式不同
2. Loss的计算不同
3. 还需要一个ref model，这样可以使得新训练的模型不偏离ref model太远

## 数据格式

一条 DPO 样本包含：

```json
{
  "chosen": [
    {"role": "user", "content": "问题"},
    {"role": "assistant", "content": "更好的回答"}
  ],
  "rejected": [
    {"role": "user", "content": "问题"},
    {"role": "assistant", "content": "更差的回答"}
  ]
}
```

## Loss 主线

对一个 batch：

```text
x_chosen.shape   = (B, L)
x_rejected.shape = (B, L)
```

拼接后：

```text
x.shape    = (2B, L)
y.shape    = (2B, L)
mask.shape = (2B, L)
```

然后：

```text
ref_logits    = ref_model(x).logits
policy_logits = policy_model(x).logits
```

logits shape：

```text
ref_logits.shape    = (2B, L, V)
policy_logits.shape = (2B, L, V)
```

取目标 token 的 log prob（按labels取出正确token的log概率）：

```text
ref_log_probs.shape    = (2B, L)
policy_log_probs.shape = (2B, L)
```

按 mask 求每条序列的总 log prob：

```text
sequence_log_probs.shape = (2B,)
```

前一半是 chosen，后一半是 rejected：

```text
chosen_log_probs.shape   = (B,)
rejected_log_probs.shape = (B,)
```

DPO 核心：

```text
pi_logratios  = chosen_policy_log_probs - rejected_policy_log_probs
ref_logratios = chosen_ref_log_probs    - rejected_ref_log_probs
logits        = pi_logratios - ref_logratios
loss          = -logsigmoid(beta * logits).mean()
```

## 需要完成的函数

### `scratch_dpo/config.py`

#### `build_dpo_data_config(...) -> DPODataConfig`

作用：

- 保存 DPO 数据阶段超参数。

输入：

- `tokenizer_dir: str`
- `data_path: str`
- `max_seq_len: int`
- `empty_think_ratio: float`

输出：

- `DPODataConfig`

#### `build_dpo_train_config(...) -> DPOTrainConfig`

作用：

- 保存 DPO 训练阶段超参数。

输入：

- `log_dir: str`
- `checkpoint_dir: str`
- `out_dir: str`
- `save_weight: str`
- `from_weight: str`
- `from_resume: str`
- `epochs: int`
- `batch_size: int`
- `learning_rate: float`
- `weight_decay: float`
- `device: str`
- `dtype: str`
- `num_workers: int`
- `accumulation_steps: int`
- `grad_clip: float`
- `log_interval: int`
- `save_interval: int`
- `warmup_steps: int`
- `min_lr_ratio: float`
- `beta: float`

输出：

- `DPOTrainConfig`

### `scratch_dpo/dataset.py`

#### `load_dpo_jsonl_records(data_path) -> list[dict[str, Any]]`

作用：

- 读取本地 DPO `jsonl` 数据。

输入：

- `data_path: str`

输出：

- `records: list[dict[str, Any]]`

每条 record 至少包含：

- `chosen: list[dict[str, Any]]`
- `rejected: list[dict[str, Any]]`

#### `build_dpo_special_token_ids(tokenizer) -> tuple[list[int], list[int]]`

作用：

- 构造 assistant span 的起止标记 token ids。

输入：

- `tokenizer`

输出：

- `assistant_bos_ids: list[int]`
- `assistant_eos_ids: list[int]`

#### `build_dpo_chat_prompt(messages, tokenizer) -> str`

作用：

- 用 chat template 把 `chosen` 或 `rejected` 转成训练文本。

输入：

- `messages: list[dict[str, Any]]`
- `tokenizer`

输出：

- `prompt_text: str`

#### `postprocess_dpo_prompt(prompt_text, empty_think_ratio=0.2) -> str`

作用：

- 对 chat template 后的文本做后处理。

输入：

- `prompt_text: str`
- `empty_think_ratio: float`

输出：

- `processed_prompt_text: str`

#### `generate_dpo_loss_mask(input_ids, assistant_bos_ids, assistant_eos_ids, max_seq_len) -> list[int]`

作用：

- 扫描 token ids，只让 assistant 回答片段参与 DPO logprob 统计。

输入：

- `input_ids: list[int]`
  - shape: `(L,)`
- `assistant_bos_ids: list[int]`
- `assistant_eos_ids: list[int]`
- `max_seq_len: int`

输出：

- `loss_mask: list[int]`
  - shape: `(L,)`

#### `build_dpo_sequence_tensors(input_ids, loss_mask, pad_token_id, max_seq_len) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`

作用：

- 把完整 token 序列整理成自回归训练需要的 `x/y/mask`。

输入：

- `input_ids: list[int]`
  - shape: `(L,)`
- `loss_mask: list[int]`
  - shape: `(L,)`
- `pad_token_id: int`
- `max_seq_len: int`

输出：

- `x: torch.Tensor`
  - shape: `(max_seq_len - 1,)`
- `y: torch.Tensor`
  - shape: `(max_seq_len - 1,)`
- `mask: torch.Tensor`
  - shape: `(max_seq_len - 1,)`

#### `build_dpo_pair_example(...) -> dict[str, torch.Tensor]`

作用：

- 把一条 `chosen/rejected` 原始样本整理成 MiniMind 风格 DPO 样本。

输入：

- `record: dict[str, Any]`
- `tokenizer`
- `assistant_bos_ids: list[int]`
- `assistant_eos_ids: list[int]`
- `max_seq_len: int`
- `empty_think_ratio: float`

输出：

- `example["x_chosen"].shape == (L,)`
- `example["y_chosen"].shape == (L,)`
- `example["mask_chosen"].shape == (L,)`
- `example["x_rejected"].shape == (L,)`
- `example["y_rejected"].shape == (L,)`
- `example["mask_rejected"].shape == (L,)`

其中：

- `L = max_seq_len - 1`

#### `DPODataset.__getitem__(idx) -> dict[str, torch.Tensor]`

作用：

- 返回一条 DPO preference pair。

输出：

- 与 `build_dpo_pair_example(...)` 一致。

### `scratch_dpo/dataloader.py`

#### `collate_dpo_batch(examples) -> dict[str, torch.Tensor]`

作用：

- 把多条 DPO 样本拼成一个 batch。

输入：

- 每条样本中所有 tensor shape 都是 `(L,)`

输出：

- batch 中所有 tensor shape 都是 `(B, L)`

#### `build_dpo_dataloader(dataset, batch_size, shuffle=True) -> DataLoader`

作用：

- 构建 DPO dataloader。

输出：

- `DataLoader`

### `scratch_dpo/loss.py`

#### `logits_to_log_probs(logits, labels) -> torch.Tensor`

作用：

- 从 logits 中取出 label 对应 token 的 log probability。

输入：

- `logits.shape == (B, L, V)`
- `labels.shape == (B, L)`

输出：

- `log_probs_per_token.shape == (B, L)`

#### `masked_sequence_log_probs(log_probs, mask) -> torch.Tensor`

作用：

- 只统计 assistant mask 位置的 log prob 总和。

输入：

- `log_probs.shape == (B, L)`
- `mask.shape == (B, L)`

输出：

- `sequence_log_probs.shape == (B,)`

#### `split_chosen_rejected(values) -> tuple[torch.Tensor, torch.Tensor]`

作用：

- 把 `(2B,)` 的值拆成 chosen 和 rejected。

输入：

- `values.shape == (2B,)`

输出：

- `chosen_values.shape == (B,)`
- `rejected_values.shape == (B,)`

#### `dpo_loss(ref_log_probs, policy_log_probs, mask, beta) -> torch.Tensor`

作用：

- 计算 DPO loss。

输入：

- `ref_log_probs.shape == (2B, L)`
- `policy_log_probs.shape == (2B, L)`
- `mask.shape == (2B, L)`
- `beta: float`

输出：

- `loss.shape == ()`

### `scratch_dpo/train_loop.py`

#### `move_dpo_batch_to_device(batch, device) -> dict[str, torch.Tensor]`

作用：

- 把 DPO batch 移动到目标设备。

输入输出：

- 所有 tensor shape 保持 `(B, L)`

#### `concat_chosen_rejected_batch(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`

作用：

- 在 batch 维度拼接 chosen 和 rejected。

输出：

- `x.shape == (2B, L)`
- `y.shape == (2B, L)`
- `mask.shape == (2B, L)`

#### `compute_dpo_train_loss(policy_model, ref_model, batch, beta) -> dict[str, torch.Tensor]`

作用：

- 跑 policy/ref 前向并计算总 loss。

输出：

- `losses["loss"].shape == ()`
- `losses["dpo_loss"].shape == ()`
- `losses["aux_loss"].shape == ()`

#### `train_dpo_one_step(...) -> float`

作用：

- 完成一次最小 DPO 优化 step。

输出：

- `loss_value: float`

#### `run_dpo_train_loop(...) -> list[float]`

作用：

- 跑最小 DPO 训练循环。

输出：

- `loss_history: list[float]`

### `scratch_dpo/train_dpo.py`

#### `build_dpo_parser() -> argparse.ArgumentParser`

作用：

- 构造正式 DPO 训练入口参数。

- `--save_weight`
- `--from_weight`
- `--from_resume`
- `--epochs`
- `--batch_size`
- `--learning_rate`
- `--accumulation_steps`
- `--grad_clip`
- `--save_interval`
- `--hidden_size`
- `--num_hidden_layers`
- `--max_seq_len`
- `--use_moe`
- `--data_path`
- `--beta`

#### `build_model_config_from_args(args) -> MiniMindConfig`

作用：

- 根据命令行参数构建模型配置。

输出：

- `MiniMindConfig`

#### `load_policy_and_reference_models(weight_path, model_config, device) -> tuple[nn.Module, nn.Module]`

作用：

- 从同一个 SFT 权重加载 policy model 和 reference model。
- `policy_model` 参与训练。
- `ref_model` 冻结，只用于提供参考 logprob。

输出：

- `policy_model`
- `ref_model`

#### `run_formal_dpo(args) -> list[float]`

作用：

- 正式 DPO 训练主入口。

输出：

- `loss_history: list[float]`

## 本地检查命令

```bash
python3 tests/run_dpo_checks.py
```

pytest 风格检查：

```bash
pytest tests/test_dpo_dataset.py tests/test_dpo_loss.py tests/test_dpo_train_loop.py
```

语法检查：

```bash
python3 -m py_compile scratch_dpo/*.py
```

## 完整运行命令

Dense DPO：

```bash
python3 scratch_dpo/train_dpo.py \
  --tokenizer_dir tokenizer \
  --data_path data/dpo.jsonl \
  --log_dir logs/dpo_dense \
  --checkpoint_dir checkpoints/dpo_dense \
  --out_dir out/dpo_dense \
  --save_weight dpo \
  --from_weight out/full_sft_dense/full_sft_768_final.pt \
  --from_resume none \
  --epochs 1 \
  --batch_size 4 \
  --learning_rate 4e-8 \
  --weight_decay 0.0 \
  --device cuda:0 \
  --dtype bfloat16 \
  --num_workers 8 \
  --accumulation_steps 1 \
  --grad_clip 1.0 \
  --log_interval 100 \
  --save_interval 100 \
  --warmup_steps 0 \
  --min_lr_ratio 0.1 \
  --beta 0.15 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --num_attention_heads 8 \
  --num_key_value_heads 2 \
  --intermediate_size 2048 \
  --max_seq_len 1024 \
  --max_position_embeddings 32768 \
  --rope_theta 1000000 \
  --rms_norm_eps 1e-6
```

MoE DPO：

```bash
python3 scratch_dpo/train_dpo.py \
  --tokenizer_dir tokenizer \
  --data_path data/dpo.jsonl \
  --log_dir logs/dpo_moe \
  --checkpoint_dir checkpoints/dpo_moe \
  --out_dir out/dpo_moe \
  --save_weight dpo \
  --from_weight out/full_sft_moe/full_sft_moe_768_moe_final.pt \
  --from_resume none \
  --epochs 1 \
  --batch_size 4 \
  --learning_rate 4e-8 \
  --weight_decay 0.0 \
  --device cuda:0 \
  --dtype bfloat16 \
  --num_workers 8 \
  --accumulation_steps 1 \
  --grad_clip 1.0 \
  --log_interval 100 \
  --save_interval 100 \
  --warmup_steps 0 \
  --min_lr_ratio 0.1 \
  --beta 0.15 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --num_attention_heads 8 \
  --num_key_value_heads 4 \
  --intermediate_size 2432 \
  --max_seq_len 1024 \
  --max_position_embeddings 32768 \
  --rope_theta 1000000 \
  --rms_norm_eps 1e-6 \
  --use_moe 1 \
  --num_experts 4 \
  --num_experts_per_tok 1 \
  --moe_intermediate_size 2432 \
  --router_aux_loss_coef 5e-4
```

## 推理

DPO 训练后的推理仍然复用 `scratch_pretrain/eval_chat.py`。

Dense DPO 推理：

```bash
python3 scratch_pretrain/eval_chat.py \
  --weight_path out/dpo_dense/dpo_768_final.pt \
  --tokenizer_dir tokenizer \
  --device cuda:0 \
  --max_new_tokens 8192 \
  --temperature 0.85 \
  --top_p 0.95 \
  --top_k 0 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --num_attention_heads 8 \
  --num_key_value_heads 2 \
  --intermediate_size 2048 \
  --max_position_embeddings 32768 \
  --rope_theta 1000000 \
  --rms_norm_eps 1e-6
```

MoE DPO 推理：

```bash
python3 scratch_pretrain/eval_chat.py \
  --weight_path out/dpo_moe/dpo_768_moe_final.pt \
  --tokenizer_dir tokenizer \
  --device cuda:0 \
  --max_new_tokens 8192 \
  --temperature 0.85 \
  --top_p 0.95 \
  --top_k 0 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --num_attention_heads 8 \
  --num_key_value_heads 4 \
  --intermediate_size 2432 \
  --max_position_embeddings 32768 \
  --rope_theta 1000000 \
  --rms_norm_eps 1e-6 \
  --use_moe \
  --num_experts 4 \
  --num_experts_per_tok 1 \
  --moe_intermediate_size 2432 \
  --router_aux_loss_coef 5e-4
```
