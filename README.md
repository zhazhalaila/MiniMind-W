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