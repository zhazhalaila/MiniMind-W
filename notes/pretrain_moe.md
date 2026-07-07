# 预训练阶段的 MoE 接入

把 Dense 版预训练主线扩展成 MoE 版预训练主线。

## 主线

```text
parse moe args
-> build moe kwargs
-> merge into model config
-> build moe model
-> forward returns lm_loss
-> collect router_aux_loss
-> combine to total_loss
-> backward
-> save dense / moe weight names
```

## 需要完成的函数

### `scratch_pretrain/moe.py`

#### `add_moe_parser_args(parser) -> argparse.ArgumentParser`

作用：

- 给正式预训练入口增加 MoE 参数。

输入：

- `parser: argparse.ArgumentParser`

输出：

- `parser: argparse.ArgumentParser`

#### `build_moe_kwargs_from_args(args) -> dict[str, Any]`

作用：

- 从命令行参数中抽出 MoE 相关字段。

输入：

- `args: argparse.Namespace`

输出：

- `moe_kwargs: dict[str, Any]`

至少包含：

- `use_moe`
- `num_experts`
- `num_experts_per_tok`
- `moe_intermediate_size`
- `router_aux_loss_coef`

#### `build_moe_weight_name(save_weight, hidden_size, use_moe) -> str`

作用：

- 统一 Dense / MoE 的权重命名。

输入：

- `save_weight: str`
- `hidden_size: int`
- `use_moe: bool`

输出：

- `weight_name: str`

#### `collect_router_aux_loss(model) -> torch.Tensor`

作用：

- 从模型中收集所有 MoE 层的 `aux_loss`。

输入：

- `model: torch.nn.Module`

输出：

- `router_aux_loss: torch.Tensor`
  - shape: `()`

#### `combine_lm_and_router_loss(lm_loss, router_aux_loss) -> torch.Tensor`

作用：

- 把主任务损失和路由辅助损失相加，得到总 loss。

输入：

- `lm_loss: torch.Tensor`
  - shape: `()`
- `router_aux_loss: torch.Tensor`
  - shape: `()`

输出：

- `total_loss: torch.Tensor`
  - shape: `()`

#### `build_moe_smoke_test_kwargs() -> dict[str, Any]`

作用：

- 构造一个本地 tiny MoE smoke test 用的小配置。

输入：

- 无

输出：

- `moe_kwargs: dict[str, Any]`

### `scratch_pretrain/train_pretrain.py`

#### `build_train_parser() -> argparse.ArgumentParser`

作用：

- 构造正式预训练入口的命令行 parser。

输入：

- 无

输出：

- `parser: argparse.ArgumentParser`

#### `build_model_config_from_args(args) -> MiniMindConfig`

作用：

- 把命令行参数整理成 MoE 版模型配置对象。

输入：

- `args: argparse.Namespace`

输出：

- `model_config: MiniMindConfig`

#### `build_runtime_from_args(args) -> dict[str, Any]`

作用：

- 一次性构造 tokenizer / dataset / dataloader / model / optimizer。

输入：

- `args: argparse.Namespace`

输出：

- `runtime: dict[str, Any]`

包含：

- `runtime["tokenizer"]`
- `runtime["dataset"]`
- `runtime["dataloader"]`
- `runtime["model"]`
- `runtime["optimizer"]`

其中：

- `runtime["dataloader"]` 迭代出的一个 batch 应当包含：
  - `batch["input_ids"]`
    - shape: `(B, L)`
  - `batch["labels"]`
    - shape: `(B, L)`

#### `run_formal_pretrain(args) -> list[float]`

作用：

- 跑正式的 MoE 预训练循环。

输入：

- `args: argparse.Namespace`

输出：

- `loss_history: list[float]`

### `model/model_minimind.py`

#### `MiniMindConfig`

作用：

- 保存 Dense / MoE 共用的模型超参数。

输入：

- 重要字段：
  - `vocab_size`
  - `hidden_size`
  - `intermediate_size`
  - `num_hidden_layers`
  - `num_attention_heads`
  - `num_key_value_heads`
  - `max_position_embeddings`
  - `rope_theta`
  - `rms_norm_eps`
  - `use_moe`
  - `num_experts`
  - `num_experts_per_tok`
  - `moe_intermediate_size`
  - `router_aux_loss_coef`

输出：

- `model_config: MiniMindConfig`

#### `MLP`

作用：

- 作为 Dense FFN，也作为 MoE expert 内部复用的 FFN 结构。

输入：

- `x: torch.Tensor`
  - shape: `(B, L, D)`

输出：

- `y: torch.Tensor`
  - shape: `(B, L, D)`

#### `MoEFeedForward`

作用：

- 实现 router + top-k experts 的 MoE FFN。

输入：

- `x: torch.Tensor`
  - shape: `(B, L, D)`

输出：

- `y: torch.Tensor`
  - shape: `(B, L, D)`

额外状态：

- `aux_loss: torch.Tensor`
  - shape: `()`

#### `DecoderLayer.forward(x, attention_mask=None) -> torch.Tensor`

作用：

- 在 decoder block 里切换 Dense FFN 和 MoE FFN。

输入：

- `x: torch.Tensor`
  - shape: `(B, L, D)`
- `attention_mask: torch.Tensor | None`
  - 期望 shape: `(1, 1, L, L)` 或可 broadcast 的等价形状

输出：

- `y: torch.Tensor`
  - shape: `(B, L, D)`

### `scratch_pretrain/entry.py`

#### `build_model(model_config, device) -> nn.Module`

作用：

- 按配置真正构造模型，并搬到目标设备上。

输入：

- `model_config: MiniMindConfig`
- `device: str | torch.device`

输出：

- `model: nn.Module`



## 本地测试命令

先跑单元检查：

```bash
python3 tests/run_pretrain_moe_checks.py
```

再跑一个本地 tiny smoke：

```bash
python3 scratch_pretrain/train_pretrain.py \
  --tokenizer_dir tokenizer \
  --data_path data/pretrain_t2t_mini.jsonl \
  --log_dir logs/local_moe_debug \
  --checkpoint_dir checkpoints/local_moe_debug \
  --out_dir out/local_moe_debug \
  --save_weight pretrain_tiny \
  --epochs 1 \
  --batch_size 2 \
  --learning_rate 1e-3 \
  --weight_decay 0.0 \
  --device cpu \
  --dtype float32 \
  --num_workers 0 \
  --accumulation_steps 1 \
  --grad_clip 1.0 \
  --log_interval 1 \
  --save_interval 100 \
  --warmup_steps 0 \
  --min_lr_ratio 1.0 \
  --hidden_size 16 \
  --num_hidden_layers 2 \
  --num_attention_heads 4 \
  --num_key_value_heads 4 \
  --intermediate_size 32 \
  --max_seq_len 8 \
  --max_position_embeddings 32 \
  --rope_theta 1000000 \
  --rms_norm_eps 1e-6 \
  --use_moe 1 \
  --num_experts 2 \
  --num_experts_per_tok 1 \
  --moe_intermediate_size 16 \
  --router_aux_loss_coef 5e-4
```

## 完整运行命令

下面这条命令按 `MiniMind` 的 `minimind-3-moe` 配置来写：

- `hidden_size = 768`
- `num_hidden_layers = 8`
- `num_attention_heads = 8`
- `num_key_value_heads = 4`
- `intermediate_size = 2432`
- `max_position_embeddings = 32768`
- `rope_theta = 1e6`
- `num_experts = 4`
- `num_experts_per_tok = 1`
- `moe_intermediate_size = 2432`
- `router_aux_loss_coef = 5e-4`
- `max_seq_len = 340`

训练侧也尽量和上游对齐：

- `epochs = 2`
- `batch_size = 32`
- `learning_rate = 5e-4`
- `weight_decay = 0.0`
- `dtype = bfloat16`
- `num_workers = 8`
- `accumulation_steps = 8`
- `grad_clip = 1.0`
- `log_interval = 100`
- `save_interval = 1000`

其中：

- `warmup_steps = 0`
- `min_lr_ratio = 1.0`

这两个是 `MiniMind-W` 额外提供的参数。这里把它们设成这个值，是为了让学习率行为尽量贴近上游的固定学习率训练方式。

```bash
python3 scratch_pretrain/train_pretrain.py \
  --tokenizer_dir tokenizer \
  --data_path data/pretrain_t2t_mini.jsonl \
  --log_dir logs/pretrain_moe \
  --checkpoint_dir checkpoints/pretrain_moe \
  --out_dir out/pretrain_moe \
  --save_weight pretrain \
  --from_weight none \
  --from_resume none \
  --epochs 2 \
  --batch_size 32 \
  --learning_rate 5e-4 \
  --weight_decay 0.0 \
  --device cuda:0 \
  --dtype bfloat16 \
  --num_workers 8 \
  --accumulation_steps 8 \
  --grad_clip 1.0 \
  --log_interval 100 \
  --save_interval 1000 \
  --warmup_steps 0 \
  --min_lr_ratio 1.0 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --num_attention_heads 8 \
  --num_key_value_heads 4 \
  --intermediate_size 2432 \
  --max_seq_len 340 \
  --max_position_embeddings 32768 \
  --rope_theta 1000000 \
  --rms_norm_eps 1e-6 \
  --use_moe 1 \
  --num_experts 4 \
  --num_experts_per_tok 1 \
  --moe_intermediate_size 2432 \
  --router_aux_loss_coef 5e-4
```

当前保存路径约定：

- 日志写到：`logs/pretrain_moe/`
- 续训断点覆盖写到：`checkpoints/pretrain_moe/pretrain_768_moe_resume_latest.pt`
- 最终权重导出到：`out/pretrain_moe/pretrain_768_moe_final.pt`

如果训练中断，恢复命令示例：

```bash
python3 scratch_pretrain/train_pretrain.py \
  --tokenizer_dir tokenizer \
  --data_path data/pretrain_t2t_mini.jsonl \
  --log_dir logs/pretrain_moe \
  --checkpoint_dir checkpoints/pretrain_moe \
  --out_dir out/pretrain_moe \
  --save_weight pretrain \
  --from_weight none \
  --from_resume checkpoints/pretrain_moe/pretrain_768_moe_resume_latest.pt \
  --epochs 2 \
  --batch_size 32 \
  --learning_rate 5e-4 \
  --weight_decay 0.0 \
  --device cuda:0 \
  --dtype bfloat16 \
  --num_workers 8 \
  --accumulation_steps 8 \
  --grad_clip 1.0 \
  --log_interval 100 \
  --save_interval 1000 \
  --warmup_steps 0 \
  --min_lr_ratio 1.0 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --num_attention_heads 8 \
  --num_key_value_heads 4 \
  --intermediate_size 2432 \
  --max_seq_len 340 \
  --max_position_embeddings 32768 \
  --rope_theta 1000000 \
  --rms_norm_eps 1e-6 \
  --use_moe 1 \
  --num_experts 4 \
  --num_experts_per_tok 1 \
  --moe_intermediate_size 2432 \
  --router_aux_loss_coef 5e-4
```
