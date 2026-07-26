"""Task heads used by the reference models."""

from __future__ import annotations

import torch
from torch import nn


class TokenClassificationHead(nn.Module):
    """Per-token classifier for MSM."""

    def __init__(self, d_model: int, num_labels: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.classifier(hidden_states)


class SequenceClassificationHead(nn.Module):
    """Mean-pooling classifier for sequence/pair classification."""

    def __init__(self, d_model: int, num_labels: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            pooled = hidden_states.mean(dim=1)
        else:
            mask = attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(self.dropout(pooled))

