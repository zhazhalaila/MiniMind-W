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

### 例子

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

![MiniMind MoE Model](figs/MiniMind_MoE_Model.jpg)

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

## KV Cache 对话推理

把对话推理链路搭起来，支持 `KV Cache`。

### 主线

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

## SFT


SFT 的核心不是改模型结构，而是改数据整理方式和 `labels` 的监督区域。

相较于Pretrain，主要区别如下：
1. 更换了数据集，SFT用的数据集是对话数据集
2. Loss的计算方式不同，在SFT数据集中用户的prompt不计入损失，只算response的Loss

### 主线

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

### 例子

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
- 但只有 `assistant` 这段会进监督，也就是说在计算Loss时，不会计算用户的prompt"杭州在哪里？"

对应的 `labels` 逻辑是：

```text
[-100, -100, ..., -100,  真正的assistant token ids..., eos_token_id, ...]
```

## DPO 

用偏好数据继续训练 SFT 模型，让模型更偏向 `chosen`，更远离 `rejected`。

与Pretrain的区别：

1. 输入的数据格式不同
2. Loss的计算不同
3. 还需要一个ref model，这样可以使得新训练的模型不偏离ref model太远

### Loss 主线

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

## LoRA

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

### 主线

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