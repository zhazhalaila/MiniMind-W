# MiniMind-W

## 预训练数据链路

```text
jsonl 文本
-> AutoTokenizer 编码
-> 在 dataset 里补 BOS / EOS
-> 在 dataset 里 pad 到固定长度
-> 在 dataset 里生成 labels
-> 得到一个训练样本
```

### 输入

- 一个 tokenizer 目录，例如 `tokenizer/`
- 一个预训练数据文件，例如 `data/pretrain_demo.jsonl`
- 一条原始样本，例如：

```json
{"text": "hello world"}
```

- 一个固定长度 `max_length`，例如 `8`

### 输出

- 单条样本输出：
  - `input_ids: torch.Tensor`
  - `labels: torch.Tensor`
- 二者的 shape 都是：

```text
(max_length,)
```

如果后面再经过 `DataLoader` 组 batch，那么 batch 后的 shape 会变成：

```text
(B, max_length)
```

## 预训练模型（Dense Model，不包含完整的训练过程，只是搭建Decoder-Only的网络架构）

![MiniMind Dense Model](figs/MiniMind_Dense_Model.jpg)

把 `input_ids` 送进一个最小的 Decoder-Only 模型前向里，并且能拿到 `logits` 和 `loss`。

主线是：

```text
input_ids
-> token embedding
-> decoder layers
-> final norm
-> lm head
-> logits
-> shifted cross entropy loss
```

```text
input_ids
-> token embedding
-> decoder layers
-> final norm
-> lm head
-> logits
-> shifted cross entropy loss
```

### 输入

- `input_ids: torch.Tensor`
  - shape: `(B, L)`
- `labels: torch.Tensor | None`
  - shape: `(B, L)`

### 输出

- `last_hidden_state`
  - shape: `(B, L, D)`
- `logits`
  - shape: `(B, L, V)`
- `loss`
  - shape: `()`

## 预训练最小训练闭环

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

## 预训练入口与 Smoke Test

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

## `train_pretrain.py`

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

### 主线

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

### 训练命令

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

## 预训练阶段的 MoE 接入

把 Dense 版预训练主线扩展成 MoE 版预训练主线。

![MiniMind Dense Model](figs/MiniMind_Dense_Model.jpg)

### 主线

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

### 训练命令

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