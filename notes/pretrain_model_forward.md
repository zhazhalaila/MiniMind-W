# 预训练模型（不适用MoE，不包含完整的训练过程，只是搭建Decoder-Only的网络架构）

这一阶段只做一件事：
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

## 输入与输出

### 输入

- `input_ids: torch.Tensor`
  - shape: `(B, L)`
- `labels: torch.Tensor | None`
  - shape: `(B, L)`
  - 只有训练时才传

### 输出

- `last_hidden_state`
  - shape: `(B, L, D)`
- `logits`
  - shape: `(B, L, V)`
- `loss`
  - shape: `()`

其中：

- `B` = batch size
- `L` = sequence length
- `D` = hidden size
- `V` = vocab size

### `MiniMindConfig`

作用：
保存模型超参数。

最重要的字段：

- `vocab_size`
- `hidden_size`
- `intermediate_size`
- `num_hidden_layers`
- `num_attention_heads`
- `num_key_value_heads`
- `max_position_embeddings`
- `rope_theta`
- `rms_norm_eps`

### `build_causal_mask(seq_len, device) -> torch.Tensor`

作用：
构造因果 mask，让当前位置只能看见自己和前面的 token。

输入：
- `seq_len: int`

输出：
- `mask.shape == (1, 1, L, L)`

### `precompute_rope_cache(...) -> tuple[torch.Tensor, torch.Tensor]`

作用：
提前算好 RoPE 需要的 `cos` / `sin` 缓存。

输出：

- `cos.shape == (max_position_embeddings, head_dim)`
- `sin.shape == (max_position_embeddings, head_dim)`

### `apply_rotary_emb(q, k, cos, sin)`

作用：
把 RoPE 应用到 `Q` 和 `K` 上。

输入：

- `q.shape == (B, Hq, L, Hd)`
- `k.shape == (B, Hk, L, Hd)`

输出：

- `q_rot.shape == q.shape`
- `k_rot.shape == k.shape`

### `RMSNorm`

作用：
归一化最后一个维度，但不改变 shape。

输入输出：

- 输入：`(B, L, D)`
- 输出：`(B, L, D)`

### `Attention`

作用：
完成一次自注意力前向。

输入输出：

- 输入：`x.shape == (B, L, D)`
- 输出：`(B, L, D)`

### `MLP`

作用：
完成一次 FFN 前向。

输入输出：

- 输入：`(B, L, D)`
- 输出：`(B, L, D)`

### `DecoderLayer`

作用：
把 Attention 和 MLP 组合成一个 decoder layer。

输入输出：

- 输入：`(B, L, D)`
- 输出：`(B, L, D)`

### `MiniMindModel`

作用：
把 token ids 变成 backbone hidden states。

输入输出：

- 输入：`input_ids.shape == (B, L)`
- 输出：`last_hidden_state.shape == (B, L, D)`

### `MiniMindForCausalLM`

作用：
在 backbone 后面接上 LM Head，并计算 shifted cross entropy。

输入输出：

- 输入：`input_ids.shape == (B, L)`
- 输出：`logits.shape == (B, L, V)`
- 如果传 `labels`，还要输出一个标量 `loss`