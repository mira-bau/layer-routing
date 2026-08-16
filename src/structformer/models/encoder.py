"""Transformer encoder backbone shared by all variants."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from structformer.models.attention import StructuralSelfAttention
from structformer.models.bias import CASABias
from structformer.models.config import TransformerConfig


@dataclass
class EncoderOutput:
    hidden_states: torch.Tensor
    attentions: list[torch.Tensor] | None = None
    structural_biases: list[torch.Tensor] | None = None


class TransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.variant = config.variant
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attention = StructuralSelfAttention(config.d_model, config.num_heads, config.dropout)
        self.dropout1 = nn.Dropout(config.dropout)

        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.d_model),
        )
        self.dropout2 = nn.Dropout(config.dropout)

        self.casa = (
            CASABias(config.d_model, config.field_vocab_size, config.casa_rank, config.casa_lambda_init)
            if config.variant == "casa"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        field_ids: torch.Tensor,
        fixed_structural_bias: torch.Tensor | None,
        need_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        attention_input = self.norm1(hidden_states)

        structural_bias = fixed_structural_bias
        if self.casa is not None:
            structural_bias = self.casa(attention_input, field_ids)

        attention_output = self.attention(
            attention_input,
            attention_mask=attention_mask,
            structural_bias=structural_bias,
            need_weights=need_weights,
        )
        hidden_states = hidden_states + self.dropout1(attention_output.hidden_states)
        hidden_states = hidden_states + self.dropout2(self.ff(self.norm2(hidden_states)))
        return hidden_states, attention_output.attention_probs, structural_bias


class TransformerEncoder(nn.Module):
    """Stack of shared encoder layers."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        field_ids: torch.Tensor,
        fixed_structural_bias: list[torch.Tensor | None] | torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> EncoderOutput:
        attentions: list[torch.Tensor] | None = [] if need_weights else None
        structural_biases: list[torch.Tensor] | None = [] if need_weights else None

        for i, layer in enumerate(self.layers):
            # Support both a single shared bias and a per-layer bias list.
            layer_bias = (
                fixed_structural_bias[i]
                if isinstance(fixed_structural_bias, list)
                else fixed_structural_bias
            )
            hidden_states, attention, structural_bias = layer(
                hidden_states,
                attention_mask=attention_mask,
                field_ids=field_ids,
                fixed_structural_bias=layer_bias,
                need_weights=need_weights,
            )
            if need_weights and attentions is not None and structural_biases is not None:
                if attention is not None:
                    attentions.append(attention)
                if structural_bias is not None:
                    structural_biases.append(structural_bias)

        return EncoderOutput(self.final_norm(hidden_states), attentions, structural_biases)

