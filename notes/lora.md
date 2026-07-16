# LoRA 阶段

在已有 `full_sft` 上，只训练少量低秩 adapter 参数，完成低成本微调。

```text
原 Linear 输出:   y = x @ W.T
加 LoRA 后输出:  y = x @ W.T + LoRA(x)
```

其中：

- 原模型权重冻结
- 只训练 LoRA 的 `A/B` 两个低秩矩阵
- 训练数据仍然是 SFT 格式
- loss 仍然是 SFT loss
- 保存时只保存 LoRA 权重，不保存完整模型

lora只工作于方阵上，也就是一个权重矩阵(in_features, out_features)，其中in_features=out_features

lora其实就是在原始model的forward后加了一层：

new_forward = model.forward(x) + LoRA(x)

其中LoRa(x) = A(x) @ B(x)，A的shape是(hidden_size, rank)，B的shape是(rank, hidden_size)

在训练时LoRA的权重矩阵的参数设定为require_grad = True, 原始model的权重矩阵的参数设定为require_grad = False

## 主线

```text
load base model
-> apply_lora
-> freeze non-lora params
-> load SFT-style data
-> forward
-> SFT loss
-> backward only LoRA params
-> save LoRA weights
```

## 需要完成的函数

### `scratch_lora/config.py`

#### `build_lora_data_config(...) -> LoRADataConfig`

作用：

- 保存 LoRA 数据阶段超参数。

输入：

- `tokenizer_dir: str`
- `data_path: str`
- `max_seq_len: int`
- `add_system_ratio: float`
- `empty_think_ratio: float`

输出：

- `LoRADataConfig`

#### `build_lora_train_config(...) -> LoRATrainConfig`

作用：

- 保存 LoRA 训练阶段超参数。

输入：

- `log_dir: str`
- `checkpoint_dir: str`
- `out_dir: str`
- `lora_name: str`
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
- `rank: int`
- `target_modules: str | None`

输出：

- `LoRATrainConfig`

### `scratch_lora/lora.py`

#### `LoRA.forward(x) -> torch.Tensor`

作用：

- 计算 LoRA 低秩分支输出。

输入：

- `x.shape == (..., in_features)`

输出：

- `output.shape == (..., out_features)`

#### `parse_target_modules(target_modules) -> list[str] | None`

作用：

- 把命令行中的 `q_proj,k_proj` 这类字符串拆成列表。

输入：

- `target_modules: str | None`

输出：

- `list[str] | None`

#### `should_apply_lora(name, module, target_modules=None, square_only=True) -> bool`

作用：

- 判断某个模块是否需要挂 LoRA。

输入：

- `name: str`
- `module: nn.Module`
- `target_modules: Iterable[str] | None`
- `square_only: bool`

输出：

- `bool`

#### `apply_lora(model, rank=16, target_modules=None, square_only=True) -> nn.Module`

作用：

- 给目标 `Linear` 层挂 LoRA 分支。

输入：

- `model: nn.Module`
- `rank: int`
- `target_modules: Iterable[str] | None`
- `square_only: bool`

输出：

- `model: nn.Module`

#### `iter_lora_modules(model) -> list[tuple[str, nn.Module]]`

作用：

- 找到已经挂了 `.lora` 的模块。

输出：

- `list[(module_name, module)]`

#### `iter_lora_parameters(model) -> list[nn.Parameter]`

作用：

- 收集 LoRA 参数。

输出：

- `list[nn.Parameter]`

#### `mark_only_lora_as_trainable(model) -> list[nn.Parameter]`

作用：

- 冻结 base model，只让 LoRA 参数参与训练。

输出：

- `lora_params: list[nn.Parameter]`

#### `save_lora(model, path) -> None`

作用：

- 只保存 LoRA 权重。

输出文件：

- `dict[str, torch.Tensor]`

#### `load_lora(model, path, device=None) -> None`

作用：

- 把 LoRA 权重加载回已经 `apply_lora` 的模型。

#### `merge_lora(model, lora_path, save_path, device=None) -> None`

作用：

- 把 LoRA 增量合并进 base Linear 权重，并导出一个完整权重。

### `scratch_lora/train_loop.py`

#### `compute_lora_sft_loss(model, batch) -> torch.Tensor`

作用：

- 跑一次前向，拿到 LoRA SFT loss。

输入：

- `batch["input_ids"].shape == (B, L)`
- `batch["labels"].shape == (B, L)`

输出：

- `loss.shape == ()`

#### `train_lora_one_step(model, batch, optimizer, device) -> float`

作用：

- 完成一次最小 LoRA 优化 step。

输出：

- `loss_value: float`

#### `run_lora_train_loop(...) -> list[float]`

作用：

- 跑最小 LoRA 训练循环。

输出：

- `loss_history: list[float]`

#### `save_lora_training_state(model, optimizer, step, path) -> None`

作用：

- 保存 LoRA 续训状态。

### `scratch_lora/train_lora.py`

#### `build_lora_parser() -> argparse.ArgumentParser`

作用：

- 构造正式 LoRA 训练入口参数。

关键参数对齐 MiniMind：

- `--lora_name`
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
- `--rank`

#### `build_model_config_from_args(args) -> MiniMindConfig`

作用：

- 根据命令行参数构建模型配置。

输出：

- `MiniMindConfig`

#### `run_formal_lora(args) -> list[float]`

作用：

- 正式 LoRA 训练主入口。

输出：

- `loss_history: list[float]`

### `scratch_lora/eval_lora.py`

#### `load_lora_inference_artifacts(...) -> tuple[tokenizer, nn.Module]`

作用：

- 加载 tokenizer、base 权重和 LoRA 权重，构造推理模型。

输入：

- `base_weight_path: str`
- `lora_weight_path: str`
- `tokenizer_dir: str`
- `model_config: MiniMindConfig`
- `device: str | torch.device`

输出：

- `tokenizer`
- `model: nn.Module`

## 本地检查命令

```bash
python3 tests/run_lora_checks.py
```

pytest 风格检查：

```bash
pytest tests/test_lora.py
```

语法检查：

```bash
python3 -m py_compile scratch_lora/*.py
```

## 完整运行命令

Dense LoRA：

```bash
python3 scratch_lora/train_lora.py \
  --tokenizer_dir tokenizer \
  --data_path data/lora_medical.jsonl \
  --log_dir logs/lora_dense \
  --checkpoint_dir checkpoints/lora_dense \
  --out_dir out/lora_dense \
  --lora_name lora_medical \
  --from_weight out/full_sft_dense/full_sft_768_final.pt \
  --from_resume none \
  --epochs 10 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --device cuda:0 \
  --dtype bfloat16 \
  --num_workers 8 \
  --accumulation_steps 1 \
  --grad_clip 1.0 \
  --log_interval 10 \
  --save_interval 1000 \
  --warmup_steps 0 \
  --min_lr_ratio 0.1 \
  --max_seq_len 340 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --num_attention_heads 8 \
  --num_key_value_heads 2 \
  --intermediate_size 2048 \
  --max_position_embeddings 32768 \
  --rope_theta 1000000 \
  --rms_norm_eps 1e-6 \
  --rank 16
```

MoE LoRA：

```bash
python3 scratch_lora/train_lora.py \
  --tokenizer_dir tokenizer \
  --data_path data/lora_medical.jsonl \
  --log_dir logs/lora_moe \
  --checkpoint_dir checkpoints/lora_moe \
  --out_dir out/lora_moe \
  --lora_name lora_medical_moe \
  --from_weight out/full_sft_moe/full_sft_moe_768_moe_final.pt \
  --from_resume none \
  --epochs 10 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --device cuda:0 \
  --dtype bfloat16 \
  --num_workers 8 \
  --accumulation_steps 1 \
  --grad_clip 1.0 \
  --log_interval 10 \
  --save_interval 1000 \
  --warmup_steps 0 \
  --min_lr_ratio 0.1 \
  --max_seq_len 340 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --num_attention_heads 8 \
  --num_key_value_heads 4 \
  --intermediate_size 2432 \
  --max_position_embeddings 32768 \
  --rope_theta 1000000 \
  --rms_norm_eps 1e-6 \
  --use_moe 1 \
  --num_experts 4 \
  --num_experts_per_tok 1 \
  --moe_intermediate_size 2432 \
  --router_aux_loss_coef 5e-4 \
  --rank 16
```

## 推理

LoRA 推理需要同时加载：

```text
base weight + lora weight
```

也可以先执行 `merge_lora(...)` 导出完整权重，再复用 `scratch_pretrain/eval_chat.py`。