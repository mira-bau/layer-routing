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


def shuffle_field_ids_for_bias(
    field_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    seed: int = 0,
) -> torch.Tensor:
    """Permute field IDs only among valid positions for the bias-content control.

    Each row's labels at positions where ``attention_mask`` is true are permuted
    among those same positions. Padding and other masked-out positions remain
    unchanged. This preserves the exact valid-position label multiset and thus
    the number and magnitude of same-group bias entries among attended token
    pairs, while destroying their alignment with token positions.

    The permutation seed is derived only from the valid labels, so a given
    example receives the same scrambled grouping independently of batch padding.
    Only the bias path is shuffled; embeddings receive the original input
    ``field_ids``, including the shared MSM mask-field ID at selected positions.
    """

    if field_ids.ndim != 2:
        raise ValueError("field_ids must have shape [batch, sequence]")
    if attention_mask.shape != field_ids.shape:
        raise ValueError("attention_mask must have the same shape as field_ids")

    batch, _ = field_ids.shape
    cpu_ids = field_ids.detach().to("cpu", dtype=torch.int64)
    cpu_mask = attention_mask.detach().to("cpu", dtype=torch.bool)
    out = cpu_ids.clone()
    for i in range(batch):
        row = cpu_ids[i]
        valid_positions = cpu_mask[i].nonzero(as_tuple=False).flatten()
        valid_ids = row[valid_positions]
        if valid_ids.numel() < 2:
            continue
        position_weights = torch.arange(1, valid_ids.numel() + 1, dtype=torch.int64)
        content_key = int((valid_ids * position_weights).sum().item())
        row_seed = (content_key ^ (int(seed) * 2654435761)) & 0x7FFFFFFFFFFFFFFF
        generator = torch.Generator().manual_seed(row_seed)
        perm = torch.randperm(valid_ids.numel(), generator=generator)
        out[i, valid_positions] = valid_ids[perm]
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
