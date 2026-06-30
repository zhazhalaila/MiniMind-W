# 预训练数据链路

```text
jsonl 文本
-> AutoTokenizer 编码
-> 在 dataset 里补 BOS / EOS
-> 在 dataset 里 pad 到固定长度
-> 在 dataset 里生成 labels
-> 得到一个训练样本
```

如果这一条链路打通了，后面模型部分只是在消费这里产出的数据。

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

## 一条样本的完整例子

假设：

- 原始文本：`"hello"`
- tokenizer 编码后：`[11, 12, 13]`
- `bos_token_id = 101`
- `eos_token_id = 102`
- `pad_token_id = 0`
- `max_length = 6`

那么整条链路会变成：

```text
原始文本
"hello"

编码后
[11, 12, 13]

补 BOS / EOS
[101, 11, 12, 13, 102]

pad 到固定长度
[101, 11, 12, 13, 102, 0]

构造 labels
[101, 11, 12, 13, 102, -100]
```