"""Self-attention with an optional structural bias on attention logits."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AttentionOutput:
    hidden_states: torch.Tensor
    attention_probs: torch.Tensor | None = None


class StructuralSelfAttention(nn.Module):
    """Multi-head self-attention with optional `[batch, seq, seq]` bias."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        structural_bias: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> AttentionOutput:
        batch_size, seq_len, _ = hidden_states.shape

        query = self._shape(self.q_proj(hidden_states), batch_size, seq_len)
        key = self._shape(self.k_proj(hidden_states), batch_size, seq_len)
        value = self._shape(self.v_proj(hidden_states), batch_size, seq_len)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if structural_bias is not None:
            scores = scores + structural_bias.unsqueeze(1).to(dtype=scores.dtype)

        if attention_mask is not None:
            key_mask = attention_mask.to(dtype=torch.bool).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~key_mask, _mask_value(scores.dtype))

        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        context = torch.matmul(probs, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(context)

        if attention_mask is not None:
            output = output * attention_mask.to(dtype=output.dtype).unsqueeze(-1)

        return AttentionOutput(output, probs if need_weights else None)

    def _shape(self, tensor: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)


def _mask_value(dtype: torch.dtype) -> float:
    if dtype in {torch.float16, torch.bfloat16}:
        return -1.0e4
    return -1.0e9

