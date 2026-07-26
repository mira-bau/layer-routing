"""Embedding stack for structured token sequences."""

from __future__ import annotations

import math

import torch
from torch import nn

from structformer.models.config import TransformerConfig


class StructuredEmbeddings(nn.Module):
    """Sum content, field, position, and optional auxiliary embeddings."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.field_embeddings = nn.Embedding(config.field_vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.max_length, config.d_model)

        self.entity_embeddings = (
            nn.Embedding(config.entity_vocab_size, config.d_model) if config.entity_vocab_size > 0 else None
        )
        self.value_type_embeddings = (
            nn.Embedding(config.value_type_vocab_size, config.d_model) if config.value_type_vocab_size > 0 else None
        )
        self.time_embeddings = nn.Embedding(config.time_vocab_size, config.d_model) if config.time_vocab_size > 0 else None

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        field_ids: torch.Tensor,
        *,
        entity_ids: torch.Tensor | None = None,
        value_type_ids: torch.Tensor | None = None,
        time_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_length:
            raise ValueError(f"Sequence length {seq_len} exceeds max_length={self.config.max_length}")

        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = (
            self.token_embeddings(input_ids)
            + self.field_embeddings(field_ids)
            + self.position_embeddings(position_ids)
        )

        if self.entity_embeddings is not None and entity_ids is not None:
            hidden = hidden + self.entity_embeddings(entity_ids)
        if self.value_type_embeddings is not None and value_type_ids is not None:
            hidden = hidden + self.value_type_embeddings(value_type_ids)
        if self.time_embeddings is not None and time_ids is not None:
            hidden = hidden + self.time_embeddings(time_ids)

        if self.config.scale_embeddings:
            hidden = hidden * math.sqrt(self.config.d_model)

        return self.dropout(hidden)
