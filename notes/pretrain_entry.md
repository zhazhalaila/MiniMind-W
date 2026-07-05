# 预训练入口与 Smoke Test

把前面已经完成的模块接成一个真正可执行的预训练入口，并且准备一个最小 smoke test。

主线是：

```text
project_root
-> build configs
-> load tokenizer
-> build dataset
-> build dataloader
-> build model
-> build optimizer
-> run_pretrain_train_loop
-> save checkpoint
-> load checkpoint
```

## 输入与输出

### 输入

- `project_root: str`
- `PretrainDataConfig`
- `PretrainTrainConfig`
- `MiniMindConfig`

### 输出

- `runtime` 字典
  - `runtime["tokenizer"]`
  - `runtime["dataset"]`
  - `runtime["dataloader"]`
  - `runtime["model"]`
  - `runtime["optimizer"]`
- `loss_history`
  - `len(loss_history) == max_steps`
- checkpoint 文件
- 读回来的 checkpoint 字典

## 例子

如果：

- `batch_size = 2`
- `max_steps = 2`
- `hidden_size = 16`
- `num_hidden_layers = 2`

那么 smoke test 的目标是：

- 真正启动一次训练入口
- 至少跑 2 个 step
- 打印 loss
- 在 `save_dir` 中生成至少 1 个 checkpoint 文件


## 需要完成的函数

### `build_model(model_config, device) -> MiniMindForCausalLM`

作用：
根据模型配置构造 `MiniMindForCausalLM`，并把模型搬到目标设备。

输入：

- `model_config: MiniMindConfig`
- `device: str | torch.device`

输出：

- `model: MiniMindForCausalLM`

### `build_pretrain_runtime(data_config, train_config, model_config) -> dict[str, Any]`

作用：
把 tokenizer、dataset、dataloader、model、optimizer 组装起来。

输出：

- `runtime["tokenizer"]`
- `runtime["dataset"]`
- `runtime["dataloader"]`
- `runtime["model"]`
- `runtime["optimizer"]`

### `run_pretrain_entry(data_config, train_config, model_config) -> list[float]`

作用：
真正执行一次最小预训练入口。

内部应该包含：

1. `build_pretrain_runtime(...)`
2. `run_pretrain_train_loop(...)`
3. 返回 `loss_history`

### `load_checkpoint_file(checkpoint_path, device) -> dict[str, Any]`

作用：
从本地文件里把 checkpoint 重新读出来。

输入：

- `checkpoint_path: str`
- `device: str | torch.device`

输出：

- `checkpoint: dict[str, Any]`

### `build_smoke_test_configs(project_root, device="cpu") -> tuple[...]`

作用：
构造一个最小 smoke test 用的 data / train / model 配置。

输入：

- `project_root: str`
- `device: str`

输出：

- `data_config: PretrainDataConfig`
- `train_config: PretrainTrainConfig`
- `model_config: MiniMindConfig`

### `run_pretrain_smoke_test(project_root, device="cpu") -> list[float]`

作用：
基于最小配置跑一个 smoke test。

输出：

- `loss_history`

### `main() -> None`

作用：
提供一个最小可执行入口。