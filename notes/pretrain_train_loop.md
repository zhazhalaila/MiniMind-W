# 预训练最小训练闭环

把前两天已经完成的 `dataset` 和 `MiniMindForCausalLM` 接起来，真正跑通最小训练闭环。

主线是：

```text
jsonl records
-> PretrainDataset
-> DataLoader
-> batch["input_ids"], batch["labels"]
-> MiniMindForCausalLM
-> logits, loss
-> backward
-> optimizer.step()
-> save checkpoint
```

## 输入与输出

### 输入

- `dataset`
  - 单条样本输出：
    - `input_ids.shape == (L,)`
    - `labels.shape == (L,)`
- `model`
  - `MiniMindForCausalLM`
- `optimizer`
  - `AdamW` 

### 输出

- 一个训练 batch
  - `batch["input_ids"].shape == (B, L)`
  - `batch["labels"].shape == (B, L)`
- 一个 step 的 `loss`
  - shape: `()`
- 一个训练循环的 `loss_history`
  - `len(loss_history) == max_steps`
- 一个 checkpoint 文件

## 例子

如果：

- `batch_size = 2`
- `max_length = 8`
- `vocab_size = 32`

那么：

- `batch["input_ids"].shape == (2, 8)`
- `batch["labels"].shape == (2, 8)`
- `logits.shape == (2, 8, 32)`
- `loss.shape == ()`



## 需要完成的函数

### `build_pretrain_train_config(...) -> PretrainTrainConfig`

作用：
保存训练阶段需要的超参数。

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

- `PretrainTrainConfig`

### `collate_pretrain_batch(examples) -> dict[str, torch.Tensor]`

作用：
把多条 `(input_ids, labels)` 样本拼成一个 batch。

输入：

- `examples`
  - 每个元素是：
    - `input_ids.shape == (L,)`
    - `labels.shape == (L,)`

输出：

- `batch["input_ids"].shape == (B, L)`
- `batch["labels"].shape == (B, L)`

### `build_pretrain_dataloader(dataset, batch_size, shuffle=True) -> DataLoader`

作用：
构造最小训练 dataloader。

输出：

- 每次迭代拿到一个 `batch`

### `build_optimizer(model, learning_rate, weight_decay) -> Optimizer`

作用：
给模型参数创建优化器。

输出：

- `torch.optim.Optimizer`

### `build_checkpoint_state(model, optimizer, step) -> dict`

作用：
把需要保存的训练状态收集到一个字典里。

输出：

- 包含：
  - `"model"`
  - `"optimizer"`
  - `"step"`

### `save_checkpoint(checkpoint, save_path) -> None`

作用：
把 checkpoint 保存到本地文件。

### `move_batch_to_device(batch, device) -> dict[str, torch.Tensor]`

作用：
把 batch 里的张量都搬到目标设备上。

输入输出 shape：

- 输入：`(B, L)`
- 输出：`(B, L)`

### `compute_pretrain_loss(model, batch) -> torch.Tensor`

作用：
只做 forward，并返回标量 loss。

输出：

- `loss.shape == ()`

### `train_one_step(model, batch, optimizer, device) -> float`

作用：
完成一次完整训练 step。

内部应该包含：

1. `model.train()`
2. `move_batch_to_device(...)`
3. `optimizer.zero_grad()`
4. forward
5. backward
6. `optimizer.step()`
7. 返回一个 python `float`

### `run_pretrain_train_loop(...) -> list[float]`

作用：
把多个 train step 串起来。

输出：

- `loss_history`
  - 预期长度：`max_steps`