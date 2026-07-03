from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import math
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class MiniMindConfig:
    """
    Minimal model config for the pretrain forward skeleton.

    Fields:
        vocab_size: int
            Input shape meaning: scalar.
            Output role: size of the LM head vocabulary dimension V.

        hidden_size: int
            Input shape meaning: scalar.
            Output role: hidden width D.

        intermediate_size: int
            Input shape meaning: scalar.
            Output role: FFN expansion width.

        num_hidden_layers: int
            Input shape meaning: scalar.
            Output role: number of decoder layers.

        num_attention_heads: int
            Input shape meaning: scalar.
            Output role: number of Q heads.

        num_key_value_heads: int
            Input shape meaning: scalar.
            Output role: number of K/V heads for GQA.

        max_position_embeddings: int
            Input shape meaning: scalar.
            Output role: RoPE cache length.

        rope_theta: float
            Input shape meaning: scalar.
            Output role: RoPE base.

        rms_norm_eps: float
            Input shape meaning: scalar.
            Output role: RMSNorm epsilon.

        head_dim: int
            Derived field.
            Output role: width of one attention head.
    """

    vocab_size: int = 6400
    hidden_size: int = 256
    intermediate_size: int = 1024
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    max_position_embeddings: int = 512
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    head_dim: int = field(init=False)

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive.")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive.")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive.")
        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive.")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive.")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive.")

        object.__setattr__(self, "head_dim", self.hidden_size // self.num_attention_heads)


@dataclass
class MiniMindModelOutput:
    """
    Backbone output container.

    Fields:
        last_hidden_state: torch.Tensor
            Shape: (B, L, D)
    """

    last_hidden_state: torch.Tensor


@dataclass
class MiniMindCausalLMOutput:
    """
    Causal LM output container.

    Fields:
        logits: torch.Tensor
            Shape: (B, L, V)

        loss: torch.Tensor | None
            Shape: () when labels are provided.
    """

    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Build a causal attention mask.

    Input:
        seq_len: int
        device: torch.device

    Output:
        causal_mask: torch.Tensor
            Shape: (1, 1, seq_len, seq_len)

    Notes:
        Expected behavior:
        - current token can see itself and previous tokens
        - current token cannot see future tokens
    """

    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.float32))
    return mask.unsqueeze(0).unsqueeze(0)


def precompute_rope_cache(
    head_dim: int,
    max_position_embeddings: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute RoPE cosine and sine caches.

    Input:
        head_dim: int
        max_position_embeddings: int
        theta: float
        device: torch.device | None

    Output:
        cos: torch.Tensor
            Shape: (max_position_embeddings, head_dim)

        sin: torch.Tensor
            Shape: (max_position_embeddings, head_dim)
    """

    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE.")
    
    # shape (head_dim // 2,), values: [theta_0, theta_1, ...]
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )

    # shape (max_position_embeddings,)
    positions = torch.arange(max_position_embeddings, device=device, dtype=torch.float32)

    # shape (max_position_embeddings, inv_freq), for each token, we have a vector of frequencies
    freqs = torch.outer(positions, inv_freq)
    # shape (max_position_embeddings, head_dim)
    # we have [theta_0, theta_1, ..., theta_0, theta_1]
    freqs = torch.cat((freqs, freqs), dim=-1)

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    return cos, sin





def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to Q and K.

    Input:
        q: torch.Tensor
            Shape: (B, Hq, L, Hd)
        k: torch.Tensor
            Shape: (B, Hk, L, Hd)
        cos: torch.Tensor
            Shape: (L, Hd) or broadcastable equivalent
        sin: torch.Tensor
            Shape: (L, Hd) or broadcastable equivalent

    Output:
        q_rot: torch.Tensor
            Shape: same as q
        k_rot: torch.Tensor
            Shape: same as k
    """

    def rotate_half(x: torch.tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot.to(q.dtype), k_rot.to(k.dtype)


class RMSNorm(nn.Module):
    """
    RMSNorm layer.

    Input to forward:
        x: torch.Tensor
            Shape: (B, L, D)

    Output from forward:
        torch.Tensor
            Shape: (B, L, D)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        out = x_float * rms
        out = out * self.weight
        return out.to(x.dtype)


class Attention(nn.Module):
    """
    Decoder self-attention block.

    Input to forward:
        x: torch.Tensor
            Shape: (B, L, D)
        attention_mask: torch.Tensor | None
            Expected shape: (1, 1, L, L) or broadcastable equivalent

    Output from forward:
        torch.Tensor
            Shape: (B, L, D)
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        if config.num_attention_heads % config.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
        
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # weight for Q (hidden_size, hidden_size)
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=False,
        )
        
        # weight for K (hidden_size, hidden_size / groups)
        # groups = attention_heads / key_value_heads
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        
        # weight for V (hidden_size, hidden_size / groups)
        # groups = attention_heads / key_value_heads
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )

        cos, sin = precompute_rope_cache(
            head_dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            theta=config.rope_theta,
        )

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=True)


    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # construct Q, K and V
        q = self.q_proj(x) # shape (B, L, Hidden_Size)
        k = self.k_proj(x) # shape (B, L, Hidden_Size / Groups) 
        v = self.v_proj(x) # shape (B, L, Hidden_Size / Groups)

        # construct multi-head
        # q shape (B, num_heads, L, head_dim) num_heads = self.num_attention_heads
        # k shape (B, num_heads/G, L, head_dim) G = self.num_attention_heads/self.num_key_value_heads
        # v shape (B, num_heads/G, L, head_dim)
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos = self.cos_cached[:seq_len].to(x.device)
        sin = self.sin_cached[:seq_len].to(x.device)

        # RoPE
        q, k = apply_rotary_emb(q, k, cos, sin)

        # extend k,v (B, num_heads/G, L, head_dim) to (B, num_heads, L, head_dim)
        if self.num_key_value_heads != self.num_attention_heads:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # attention score shape (B, num_heads, L, L)
        # each element represents attention score
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        causal_mask = build_causal_mask(seq_len, x.device)
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        # softmax on masked matrix, shape (B, num_heads, L, L)
        attn_weights = F.softmax(scores.float(), dim=-1).to(q.dtype)

        # output shape (B, num_heads, L, head_dim)
        output = torch.matmul(attn_weights, v)
        # shape (B, L, num_heads, head_dim)
        output = output.transpose(1, 2).contiguous()
        # shape (B, L, Hidden_Size)
        output = output.view(batch_size, seq_len, self.num_attention_heads * self.head_dim)

        output = self.o_proj(output)
        return output

class MLP(nn.Module):
    """
    Feed-forward block.

    Input to forward:
        x: torch.Tensor
            Shape: (B, L, D)

    Output from forward:
        torch.Tensor
            Shape: (B, L, D)
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config

        # gate unit
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False
        )

        # up dimension
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False
        )

        # down dimension
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gate shape (B, L, Intermediate_Size)
        gate = F.silu(self.gate_proj(x))
        # up shape (B, L, Intermediate_Size)
        up = self.up_proj(x)
        # hidden shape (B, L, Intermediate_Size)
        hidden = gate * up
        # out shape (B, L, Hidden_Size)
        out = self.down_proj(hidden)
        return out


class DecoderLayer(nn.Module):
    """
    One decoder layer = attention sublayer + FFN sublayer.

    Input to forward:
        x: torch.Tensor
            Shape: (B, L, D)
        attention_mask: torch.Tensor | None
            Expected shape: (1, 1, L, L) or broadcastable equivalent

    Output from forward:
        torch.Tensor
            Shape: (B, L, D)
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

        self.self_attn = Attention(config)

        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        Attention Input Shape (B, L, D)
                  Output Shape (B, L, D)
        MLP Input Shape (B, L, D)
            Output Shape (B, L, D)
        '''
        
        attn_input = self.input_layernorm(x)
        attn_output = self.self_attn(attn_input, attention_mask=attention_mask)
        x = x + attn_output

        mlp_input = self.post_attention_layernorm(x)
        mlp_output = self.mlp(mlp_input)
        x = x + mlp_output

        return x


class MiniMindModel(nn.Module):
    """
    Decoder-only backbone.

    Input to forward:
        input_ids: torch.Tensor
            Shape: (B, L)
        attention_mask: torch.Tensor | None
            Optional external mask

    Output from forward:
        MiniMindModelOutput
            last_hidden_state shape: (B, L, D)
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> MiniMindModelOutput:
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        hidden_states = self.norm(hidden_states)

        return MiniMindModelOutput(last_hidden_state=hidden_states)

class MiniMindForCausalLM(nn.Module):
    """
    Causal LM wrapper for backbone + LM head + shifted loss.

    Input to forward:
        input_ids: torch.Tensor
            Shape: (B, L)
        labels: torch.Tensor | None
            Shape: (B, L)
        attention_mask: torch.Tensor | None

    Output from forward:
        MiniMindCausalLMOutput
            logits shape: (B, L, V)
            loss shape: () when labels are provided
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config

        self.model = MiniMindModel(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> MiniMindCausalLMOutput:
        
        # shape (B, L, Hidden_Size)
        model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = model_output.last_hidden_state
        
        # shape (B, L, vocab_size)
        # for each token of L, we have vocab size candidate tokens 
        logits = self.lm_head(hidden_states)

        loss = None

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return MiniMindCausalLMOutput(
            logits=logits,
            loss=loss,
        )
