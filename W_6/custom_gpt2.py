"""Minimal GPT-2 decoder implementation for weight parity checks."""
import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    """HuggingFace GELU approximation used by GPT-2."""
    # Matches the tanh-based GELU variant in the original GPT-2 implementation.
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


@dataclass
class GPT2Config:
    """Subset of GPT-2 configuration values needed for forward pass."""
    vocab_size: int = 50257
    n_positions: int = 1024
    n_ctx: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: Optional[int] = None
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    bos_token_id: int = 50256
    eos_token_id: int = 50256

    def __post_init__(self):
        if self.n_inner is None:
            self.n_inner = 4 * self.n_embd


class GPT2SelfAttention(nn.Module):
    """Single GPT-2 self-attention block (weights align with HuggingFace GPT2Attention)."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dim must be divisible by num heads"
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale_attn = 1.0 / math.sqrt(self.head_dim)

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        bias = torch.tril(torch.ones((config.n_positions, config.n_positions), dtype=torch.bool))
        self.register_buffer("bias", bias.view(1, 1, config.n_positions, config.n_positions), persistent=False)
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.size()
        qkv = self.c_attn(hidden_states)
        # HF packs q, k, v into a single projection; split and reshape per head.
        query, key, value = qkv.split(qkv.size(-1) // 3, dim=-1)

        query = query.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scale_attn
        causal_mask = self.bias[:, :, :seq_len, :seq_len]
        # Prevent attention to future positions using the cached causal mask.
        attn_weights = attn_weights.masked_fill(~causal_mask, self.masked_bias)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_probs = self.attn_dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output


class GPT2MLP(nn.Module):
    """GPT-2 feed-forward block (matches HuggingFace naming)."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        inner_dim = config.n_inner
        self.c_fc = nn.Linear(config.n_embd, inner_dim, bias=True)
        self.c_proj = nn.Linear(inner_dim, config.n_embd, bias=True)
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.act = gelu_new if config.activation_function == "gelu_new" else F.gelu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    """Transformer decoder block with pre-norm residual structure."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_input = self.ln_1(hidden_states)
        attn_output = self.attn(attn_input, attention_mask=attention_mask)
        hidden_states = hidden_states + attn_output
        mlp_input = self.ln_2(hidden_states)
        mlp_output = self.mlp(mlp_input)
        hidden_states = hidden_states + mlp_output
        return hidden_states


class GPT2Model(nn.Module):
    """Stack of decoder blocks with token & position embeddings."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        if attention_mask is not None:
            # Convert binary mask (1 keep, 0 pad) into additive bias that matches HF logic.
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -1e4

        for block in self.h:
            hidden_states = block(hidden_states, attention_mask=attention_mask)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class GPT2LMHeadModel(nn.Module):
    """Language modeling head that mirrors HuggingFace GPT2LMHeadModel."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # GPT-2 ties token embedding weights with LM head for faster convergence.
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.transformer(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)
        return logits
