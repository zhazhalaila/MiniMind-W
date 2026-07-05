# `train_pretrain.py`

把下面这些能力接起来：

- 读取命令行参数
- 构造 tokenizer / dataset / dataloader / model / optimizer
- 运行正式 train loop
- 支持 gradient accumulation
- 支持 grad clipping
- 支持 warmup + cosine learning rate decay
- 支持 `float16` 下的 `GradScaler`
- 支持 checkpoint resume
- 把日志写到 `logs/`
- 把续训断点写到 `checkpoints/`
- 把最终权重导出到 `out/`

## 主线

```text
parse args
-> build runtime
-> optional load weight
-> optional resume checkpoint
-> build autocast / grad scaler
-> training loop
   -> forward
   -> backward
   -> accumulation
   -> grad clipping
   -> optimizer step
   -> lr scheduling
   -> logging
   -> checkpoint save
-> final checkpoint save
-> final weight export
```

## 目录约定

### `logs/`

存训练日志：

- 文本日志
  - 例如：`logs/pretrain.log`
- 结构化指标
  - 例如：`logs/pretrain_metrics.jsonl`

### `checkpoints/`

存中途断点，用于 resume：

- 例如：`checkpoints/step_1000.pt`
- 例如：`checkpoints/pretrain_last.pt`

### `out/`

存最终导出的模型权重：

- 例如：`out/pretrain_final.pt`

## 当前命令行参数

### 路径相关

- `--tokenizer_dir`
- `--data_path`
- `--log_dir`
- `--checkpoint_dir`
- `--out_dir`

### 加载相关

- `--save_weight`
- `--from_weight`
- `--from_resume`

### 训练相关

- `--epochs`
- `--batch_size`
- `--learning_rate`
- `--weight_decay`
- `--device`
- `--dtype`
- `--num_workers`
- `--accumulation_steps`
- `--grad_clip`
- `--log_interval`
- `--save_interval`
- `--warmup_steps`
- `--min_lr_ratio`

### 模型相关

- `--hidden_size`
- `--num_hidden_layers`
- `--num_attention_heads`
- `--num_key_value_heads`
- `--intermediate_size`
- `--max_seq_len`

## 关键函数与当前作用

### `build_train_parser() -> argparse.ArgumentParser`

作用：

- 定义正式预训练入口的全部参数

### `parse_train_args(argv=None) -> argparse.Namespace`

作用：

- 解析命令行参数
- 做基本合法性校验

当前会检查：

- `epochs > 0`
- `batch_size > 0`
- `accumulation_steps > 0`
- `num_workers >= 0`
- `warmup_steps >= 0`
- `0 <= min_lr_ratio <= 1`
- `from_resume` 和 `from_weight` 不能同时设置

### `build_data_config_from_args(args) -> PretrainDataConfig`

作用：

- 把命令行里的数据相关参数映射成 `PretrainDataConfig`

当前映射关系：

- `args.tokenizer_dir -> tokenizer_dir`
- `args.data_path -> data_path`
- `args.max_seq_len -> max_length`

### `build_train_config_from_args(args) -> PretrainTrainConfig`

作用：

- 把命令行里的训练参数映射成 `PretrainTrainConfig`

注意：

- 这个配置对象主要还是兼容之前拆分出来的底层模块
- 正式 train loop 自己还会额外直接读取：
  - `accumulation_steps`
  - `grad_clip`
  - `warmup_steps`
  - `min_lr_ratio`
  - `from_resume`

### `build_model_config_from_args(args) -> MiniMindConfig`

作用：

- 用命令行参数构造模型配置

注意：

- 这个函数默认 `vocab_size=6400`
- 真正正式训练时，`build_runtime_from_args()` 会按 `len(tokenizer)` 重建一次 model config，使模型词表大小和 tokenizer 对齐

### `build_runtime_from_args(args) -> dict[str, Any]`

作用：

- 一次性构造：
  - tokenizer
  - dataset
  - dataloader
  - model
  - optimizer

输出字典包含：

- `runtime["tokenizer"]`
- `runtime["dataset"]`
- `runtime["dataloader"]`
- `runtime["model"]`
- `runtime["optimizer"]`

### `build_autocast_context(device, dtype) -> Any`

作用：

- 构造混合精度上下文

当前行为：

- `cpu` -> `nullcontext()`
- `cuda + float16` -> `torch.autocast(..., torch.float16)`
- `cuda + bfloat16` -> `torch.autocast(..., torch.bfloat16)`
- `cuda + float32` -> `nullcontext()`

### `build_grad_scaler(device, dtype) -> GradScaler`

作用：

- 为 `float16` 训练准备梯度缩放器

当前行为：

- 只有 `cuda + float16` 时启用
- 其余情况自动关闭

### `compute_learning_rate(current_step, total_steps, base_learning_rate, warmup_steps, min_lr_ratio) -> float`

作用：

- 计算当前 update step 的学习率

当前策略：

- 前 `warmup_steps` 线性升温
- 后续 cosine decay
- 最低衰减到：

```text
base_learning_rate * min_lr_ratio
```

### `set_optimizer_learning_rate(optimizer, learning_rate) -> None`

作用：

- 把计算好的学习率写回 optimizer

### `format_train_log(step, loss, learning_rate) -> str`

作用：

- 把一条训练记录格式化成文本日志

当前格式：

```text
step=... loss=... lr=...
```

### `append_train_metric(metrics_path, step, loss, learning_rate) -> None`

作用：

- 往 `jsonl` 文件追加一条结构化指标

当前字段：

- `step`
- `loss`
- `learning_rate`

### `load_training_state(checkpoint_path, model, optimizer, device) -> int`

作用：

- 从 checkpoint 恢复：
  - model
  - optimizer
  - step

注意：

- 当前 formal loop 里 resume 逻辑已经直接展开了
- 所以这个函数现在更像一个底层辅助函数

### `_build_formal_checkpoint_state(...) -> dict`

作用：

- 构造正式 checkpoint 字典

当前包含字段：

- `model`
- `optimizer`
- `step`
- `epoch`
- `batch_in_epoch`
- `scaler`（如果存在）

### `_build_epoch_dataloader(...)`

作用：

- 每个 epoch 单独构造 dataloader
- 用固定 seed 让 shuffle 顺序可复现
- 这样 resume 时才能跳过前面已经训练过的 batch

### `run_formal_pretrain(args) -> list[float]`

作用：

- 正式预训练主入口

当前内部流程：

1. 创建 `logs/`、`checkpoints/`、`out/`
2. 构造 runtime
3. 可选加载 `from_weight`
4. 可选 resume `from_resume`
5. 创建 autocast context 和 grad scaler
6. 按 epoch 训练
7. 每次 update 前根据 scheduler 设置学习率
8. forward + backward
9. 按 `accumulation_steps` 进行梯度累积
10. step 前做 grad clipping
11. 写日志与 metrics
12. 按 `save_interval` 保存 checkpoint
13. 训练结束后再保存：
    - `checkpoints/{save_weight}_last.pt`
    - `out/{save_weight}_final.pt`

### `main(argv=None) -> None`

作用：

- 命令行入口

当前行为：

```text
parse_train_args
-> run_formal_pretrain
```

## 本地小规模命令示例（本机配置较低的情况下可以先看看有没有报错）

```bash
python3 scratch_pretrain/train_pretrain.py \
  --tokenizer_dir tokenizer \
  --data_path data/pretrain_t2t_mini.jsonl \
  --log_dir logs/local_debug \
  --checkpoint_dir checkpoints/local_debug \
  --out_dir out/local_debug \
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
  --save_interval 5000 \
  --warmup_steps 0 \
  --min_lr_ratio 0.1 \
  --hidden_size 16 \
  --num_hidden_layers 2 \
  --num_attention_heads 4 \
  --num_key_value_heads 4 \
  --intermediate_size 32 \
  --max_seq_len 8
```

## 训练命令（如果配置较高）

```bash
python3 scratch_pretrain/train_pretrain.py \
  --tokenizer_dir tokenizer \
  --data_path data/pretrain_t2t_mini.jsonl \
  --log_dir logs/pretrain_dense \
  --checkpoint_dir checkpoints/pretrain_dense \
  --out_dir out/pretrain_dense \
  --save_weight pretrain_dense \
  --epochs 1 \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --weight_decay 0.1 \
  --device cuda:0 \
  --dtype bfloat16 \
  --num_workers 8 \
  --accumulation_steps 8 \
  --grad_clip 1.0 \
  --log_interval 100 \
  --save_interval 1000 \
  --warmup_steps 1000 \
  --min_lr_ratio 0.1 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --num_attention_heads 8 \
  --num_key_value_heads 2 \
  --intermediate_size 2048 \
  --max_seq_len 512
```

当前保存策略：

- 日志写到：`logs/pretrain_dense/`
- 续训断点覆盖写到：`checkpoints/pretrain_dense/pretrain_dense_resume_latest.pt`
- 最终权重导出到：`out/pretrain_dense/pretrain_dense_final.pt`

如果训练中断，恢复命令示例：

```bash
python3 scratch_pretrain/train_pretrain.py \
  --tokenizer_dir tokenizer \
  --data_path data/pretrain_t2t_mini.jsonl \
  --log_dir logs/pretrain_dense \
  --checkpoint_dir checkpoints/pretrain_dense \
  --out_dir out/pretrain_dense \
  --save_weight pretrain_dense \
  --from_resume checkpoints/pretrain_dense/pretrain_dense_resume_latest.pt \
  --epochs 1 \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --weight_decay 0.1 \
  --device cuda:0 \
  --dtype bfloat16 \
  --num_workers 8 \
  --accumulation_steps 8 \
  --grad_clip 1.0 \
  --log_interval 100 \
  --save_interval 1000 \
  --warmup_steps 1000 \
  --min_lr_ratio 0.1 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --num_attention_heads 8 \
  --num_key_value_heads 2 \
  --intermediate_size 2048 \
  --max_seq_len 512
```