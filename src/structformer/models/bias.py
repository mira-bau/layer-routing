"""Structural attention bias modules."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SAABWeights:
    field: float = 1.0
    entity: float = 1.0
    value_type: float = 0.3
    time: float = 0.5


def _same_tag_bias(ids: torch.Tensor, weight: float) -> torch.Tensor:
    if weight == 0:
        return torch.zeros(ids.shape[0], ids.shape[1], ids.shape[1], device=ids.device, dtype=torch.float32)
    same = ids.unsqueeze(2).eq(ids.unsqueeze(1))
    return same.to(dtype=torch.float32) * weight


def shuffle_field_ids_for_bias(field_ids: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Permute field_ids within each sequence for the bias-content control.

    Returns a tensor with the same shape as ``field_ids`` in which each row is a
    permutation of that row's labels. This preserves the exact label multiset per
    sequence---so the derived bias has the same number of same-group pairs and the
    same magnitude as the true SAAB bias---while destroying the alignment between
    token position and true field. The permutation is seeded from each row's
    contents, so a given example always yields the same scrambled grouping at both
    train and diagnostic time.

    Only the bias is affected; embeddings continue to receive the true field_ids,
    so no structural information or capacity is added or removed.
    """

    batch, seq = field_ids.shape
    cpu_ids = field_ids.detach().to("cpu", dtype=torch.int64)
    position_weights = torch.arange(1, seq + 1, dtype=torch.int64)
    out = torch.empty_like(cpu_ids)
    for i in range(batch):
        row = cpu_ids[i]
        content_key = int((row * position_weights).sum().item())
        row_seed = (content_key ^ (int(seed) * 2654435761)) & 0x7FFFFFFFFFFFFFFF
        generator = torch.Generator().manual_seed(row_seed)
        perm = torch.randperm(seq, generator=generator)
        out[i] = row[perm]
    return out.to(device=field_ids.device, dtype=field_ids.dtype)


def build_saab_bias(
    field_ids: torch.Tensor,
    *,
    entity_ids: torch.Tensor | None = None,
    value_type_ids: torch.Tensor | None = None,
    time_ids: torch.Tensor | None = None,
    weights: SAABWeights | None = None,
) -> torch.Tensor:
    """Build the fixed SAAB pairwise structural bias.

    Inputs are integer tag tensors with shape `[batch, seq]`. The returned bias
    has shape `[batch, seq, seq]` and is broadcast across attention heads later.
    """

    weights = weights or SAABWeights()
    bias = _same_tag_bias(field_ids, weights.field)
    if entity_ids is not None:
        bias = bias + _same_tag_bias(entity_ids, weights.entity)
    if value_type_ids is not None:
        bias = bias + _same_tag_bias(value_type_ids, weights.value_type)
    if time_ids is not None:
        bias = bias + _same_tag_bias(time_ids, weights.time)
    return bias.to(device=field_ids.device)
