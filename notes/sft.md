# SFT 阶段


SFT 的核心不是改模型结构，而是改数据整理方式和 `labels` 的监督区域。

## 主线

```text
record["conversations"]
-> format_sft_messages(...)
-> build_sft_chat_prompt(...)
-> postprocess_sft_prompt(...)
-> tokenize
-> input_ids
-> build_sft_special_token_ids(...)
-> generate_sft_labels(...)
-> pad
-> batch
-> model(input_ids, labels)
-> sft loss
```

## 例子

假设 chat template 后的文本逻辑是：

```text
<bos>user
杭州在哪里？
<eos>
<bos>assistant
杭州在浙江省。
<eos>
```

那么：

- 整段都会进 `input_ids`
- 但只有 `assistant` 这段会进监督

对应的 `labels` 逻辑是：

```text
[-100, -100, ..., -100,  真正的assistant token ids..., eos_token_id, ...]
```

### `scratch_sft/config.py`

#### `build_sft_data_config(...) -> SFTDataConfig`

作用：

- 保存 SFT 数据阶段需要的超参数。

输入：

- `tokenizer_dir: str`
- `data_path: str`
- `max_seq_len: int`
- `add_system_ratio: float`
- `empty_think_ratio: float`

输出：

- `SFTDataConfig`

#### `build_sft_train_config(...) -> SFTTrainConfig`

作用：

- 保存 SFT 训练阶段需要的超参数。

输入：

- `save_dir: str`
- `batch_size: int`
- `learning_rate: float`
- `weight_decay: float`
- `max_steps: int`
- `device: str`
- `log_every: int`
- `save_every: int`

输出：

- `SFTTrainConfig`

### `scratch_sft/prompt.py`

#### `format_sft_messages(record, add_system_ratio=0.2) -> list[dict[str, Any]]`

作用：

- 把原始 `record["conversations"]` 整理成 `messages`。

输入：

- `record: dict[str, Any]`
  - 期望 `record["conversations"]` 是一个多轮列表
- `add_system_ratio: float`

输出：

- `messages: list[dict[str, Any]]`
  - 每个元素至少有：
    - `role: str`
    - `content: str`
  - 也可能保留：
    - `reasoning_content`
    - `tools`
    - `tool_calls`

#### `build_sft_chat_prompt(messages, tokenizer) -> str`

作用：

- 用 `tokenizer.apply_chat_template(...)` 构造整段 SFT 训练文本。

输入：

- `messages: list[dict[str, Any]]`
- `tokenizer`

输出：

- `prompt_text: str`

#### `postprocess_sft_prompt(prompt_text, empty_think_ratio=0.2) -> str`

作用：

- 对 chat template 后的文本做后处理。

输入：

- `prompt_text: str`
- `empty_think_ratio: float`

输出：

- `processed_prompt_text: str`

#### `build_sft_special_token_ids(tokenizer) -> tuple[list[int], list[int]]`

作用：

- 预先构造 assistant span 的起止标记 token ids。

输入：

- `tokenizer`

输出：

- `assistant_bos_ids: list[int]`
- `assistant_eos_ids: list[int]`

#### `generate_sft_labels(input_ids, assistant_bos_ids, assistant_eos_ids, max_seq_len) -> list[int]`

作用：

- 扫描 `input_ids`，只让 assistant 回答片段参与 loss。

输入：

- `input_ids: list[int]`
  - shape: `(L,)`
- `assistant_bos_ids: list[int]`
- `assistant_eos_ids: list[int]`
- `max_seq_len: int`

输出：

- `labels: list[int]`
  - shape: `(L,)`

#### `pad_sft_example(input_ids, labels, pad_token_id, max_seq_len) -> tuple[list[int], list[int]]`

作用：

- 把单条样本 pad / truncate 到固定长度。

输入：

- `input_ids: list[int]`
  - shape: `(L,)`
- `labels: list[int]`
  - shape: `(L,)`
- `pad_token_id: int`
- `max_seq_len: int`

输出：

- `padded_input_ids: list[int]`
  - shape: `(max_seq_len,)`
- `padded_labels: list[int]`
  - shape: `(max_seq_len,)`

### `scratch_sft/dataset.py`

#### `load_sft_jsonl_records(data_path) -> list[dict[str, Any]]`

作用：

- 读取本地 `jsonl` 的 SFT 数据。

输入：

- `data_path: str`

输出：

- `records: list[dict[str, Any]]`

#### `build_sft_example(record, tokenizer, assistant_bos_ids, assistant_eos_ids, max_seq_len, add_system_ratio=0.2, empty_think_ratio=0.2) -> dict[str, torch.Tensor]`

作用：

- 把一条原始 `conversations` 样本整理成训练样本。

输入：

- `record: dict[str, Any]`
- `tokenizer`
- `assistant_bos_ids: list[int]`
- `assistant_eos_ids: list[int]`
- `max_seq_len: int`
- `add_system_ratio: float`
- `empty_think_ratio: float`

输出：

- `example["input_ids"]: torch.Tensor`
  - shape: `(max_seq_len,)`
- `example["labels"]: torch.Tensor`
  - shape: `(max_seq_len,)`

#### `SFTDataset.__getitem__(idx) -> dict[str, torch.Tensor]`

作用：

- 返回一条 SFT 训练样本。

输出：

- `input_ids.shape == (L,)`
- `labels.shape == (L,)`

### `scratch_sft/dataloader.py`

#### `collate_sft_batch(examples) -> dict[str, torch.Tensor]`

作用：

- 把多条 SFT 样本拼成一个 batch。

输入：

- 每条样本：
  - `input_ids.shape == (L,)`
  - `labels.shape == (L,)`

输出：

- `batch["input_ids"].shape == (B, L)`
- `batch["labels"].shape == (B, L)`

#### `build_sft_dataloader(dataset, batch_size, shuffle=True) -> DataLoader`

作用：

- 构造 SFT dataloader。

### `scratch_sft/train_loop.py`

#### `compute_sft_loss(model, batch) -> torch.Tensor`

作用：

- 跑一次前向，拿到 SFT 标量 loss。

输入：

- `batch["input_ids"].shape == (B, L)`
- `batch["labels"].shape == (B, L)`

输出：

- `loss: torch.Tensor`
  - shape: `()`

#### `train_sft_one_step(model, batch, optimizer, device) -> float`

作用：

- 完成一次完整的 SFT 训练 step。

输出：

- `loss_value: float`

#### `run_sft_train_loop(...) -> list[float]`

作用：

- 跑 SFT 训练循环。

输出：

- `loss_history: list[float]`

### `scratch_sft/train_sft.py`

#### `build_sft_parser() -> argparse.ArgumentParser`

作用：

- 构造正式 SFT 训练入口参数。

#### `build_model_config_from_args(args) -> MiniMindConfig`

作用：

- 把命令行参数整理成模型配置。

#### `run_formal_sft(args) -> list[float]`

作用：

- 跑正式 SFT 训练主入口。

## 本地测试命令

先跑单元检查：

```bash
python3 tests/run_sft_checks.py
```

如果你补完实现，还要跑一个本地小规模检查：

```bash
python3 scratch_sft/entry.py
```

## 完整运行命令

Dense Model SFT

```bash
python3 scratch_sft/train_sft.py \
  --tokenizer_dir tokenizer \
  --data_path data/sft_t2t_mini.jsonl \
  --log_dir logs/full_sft_dense \
  --checkpoint_dir checkpoints/full_sft_dense \
  --out_dir out/full_sft_dense \
  --save_weight full_sft \
  --from_weight out/pretrain_dense/pretrain_dense_final.pt \
  --from_resume none \
  --epochs 2 \
  --batch_size 16 \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --device cuda:0 \
  --dtype bfloat16 \
  --num_workers 8 \
  --accumulation_steps 1 \
  --grad_clip 1.0 \
  --log_interval 100 \
  --save_interval 1000 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --num_attention_heads 8 \
  --num_key_value_heads 2 \
  --intermediate_size 2048 \
  --max_position_embeddings 32768 \
  --rope_theta 1000000 \
  --rms_norm_eps 1e-6 \
  --max_seq_len 768 \
  --add_system_ratio 0.2 \
  --empty_think_ratio 0.2
```

MoE Model SFT

```bash
python3 scratch_sft/train_sft.py \
  --tokenizer_dir tokenizer \
  --data_path data/sft_t2t_mini.jsonl \
  --log_dir logs/full_sft_moe \
  --checkpoint_dir checkpoints/full_sft_moe \
  --out_dir out/full_sft_moe \
  --save_weight full_sft_moe \
  --from_weight out/pretrain_moe/pretrain_768_moe_final.pt \
  --from_resume none \
  --epochs 2 \
  --batch_size 16 \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --device cuda:0 \
  --dtype bfloat16 \
  --num_workers 8 \
  --accumulation_steps 1 \
  --grad_clip 1.0 \
  --log_interval 100 \
  --save_interval 1000 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --num_attention_heads 8 \
  --num_key_value_heads 4 \
  --intermediate_size 2432 \
  --max_position_embeddings 32768 \
  --rope_theta 1000000 \
  --rms_norm_eps 1e-6 \
  --max_seq_len 768 \
  --add_system_ratio 0.2 \
  --empty_think_ratio 0.2 \
  --use_moe 1 \
  --num_experts 4 \
  --num_experts_per_tok 1 \
  --moe_intermediate_size 2432 \
  --router_aux_loss_coef 5e-4
```

## 推理命令

Dense Model Inference
```
python3 scratch_pretrain/eval_chat.py \
  --weight_path out/full_sft_dense/full_sft_768_final.pt \
  --tokenizer_dir tokenizer \
  --device cuda:0 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --num_attention_heads 8 \
  --num_key_value_heads 2 \
  --intermediate_size 2048 \
  --max_position_embeddings 32768 \
  --rope_theta 1000000 \
  --rms_norm_eps 1e-6 \
  --max_new_tokens 512 \
  --temperature 0.85 \
  --top_p 0.90 \
  --top_k 50
```

MoE Model Inference
```
python3 scratch_pretrain/eval_chat.py \
  --weight_path out/full_sft_moe/full_sft_moe_768_moe_final.pt \
  --tokenizer_dir tokenizer \
  --device cuda:0 \
  --use_moe \
  --num_experts 4 \
  --num_experts_per_tok 1 \
  --moe_intermediate_size 2432 \
  --router_aux_loss_coef 5e-4 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --num_attention_heads 8 \
  --num_key_value_heads 4 \
  --intermediate_size 2432 \
  --max_position_embeddings 32768 \
  --rope_theta 1000000 \
  --rms_norm_eps 1e-6 \
  --max_new_tokens 512 \
  --temperature 0.85 \
  --top_p 0.90 \
  --top_k 50
```
