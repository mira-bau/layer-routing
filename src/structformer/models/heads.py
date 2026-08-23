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
