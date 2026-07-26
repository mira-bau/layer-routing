"""Masked Structure Modeling helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


MSM_IGNORE_INDEX = -100


@dataclass(frozen=True)
class MSMBatch:
    input_ids: torch.Tensor
    field_ids: torch.Tensor
    attention_mask: torch.Tensor


@dataclass(frozen=True)
class MaskedMSMBatch:
    input_ids: torch.Tensor
    field_ids: torch.Tensor
    original_field_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    mask_positions: torch.Tensor


def mask_field_ids(
    field_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    mask_field_id: int,
    mask_probability: float = 0.15,
    ignore_index: int = MSM_IGNORE_INDEX,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask field IDs and return masked IDs, labels, and mask positions.

    Token IDs are intentionally untouched. Labels use `ignore_index` everywhere
    except selected field positions.
    """

    if not 0.0 < mask_probability <= 1.0:
        raise ValueError("mask_probability must be in the interval (0, 1]")

    eligible = attention_mask.to(dtype=torch.bool)
    mask_positions = torch.zeros_like(eligible, dtype=torch.bool)

    for field_id in torch.unique(field_ids[eligible]):
        field_positions = torch.nonzero(eligible & field_ids.eq(field_id), as_tuple=False)
        if field_positions.numel() == 0:
            continue
        budget = max(1, int(round(field_positions.shape[0] * mask_probability)))
        budget = min(budget, field_positions.shape[0])
        order = torch.randperm(field_positions.shape[0], generator=generator, device=field_ids.device)
        selected = field_positions[order[:budget]]
        mask_positions[selected[:, 0], selected[:, 1]] = True

    masked_field_ids = field_ids.clone()
    masked_field_ids[mask_positions] = mask_field_id

    labels = torch.full_like(field_ids, ignore_index)
    labels[mask_positions] = field_ids[mask_positions]
    return masked_field_ids, labels, mask_positions


def msm_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, *, ignore_index: int = MSM_IGNORE_INDEX) -> torch.Tensor:
    """Mean CE over masked field positions."""

    if labels.ne(ignore_index).sum().item() == 0:
        raise ValueError("MSM loss requires at least one masked position")
    return F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=ignore_index)


def make_synthetic_msm_batch(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    title_field_id: int = 4,
    content_field_id: int = 3,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> MSMBatch:
    """Create a tiny synthetic batch where token ranges correlate with fields."""

    if vocab_size < 12:
        raise ValueError("vocab_size must be at least 12 for the synthetic MSM batch")
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")

    split = seq_len // 2
    field_ids = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
    field_ids[:, :split] = title_field_id
    field_ids[:, split:] = content_field_id

    midpoint = max(6, vocab_size // 2)
    title_tokens = torch.randint(2, midpoint, (batch_size, split), generator=generator, device=device)
    content_tokens = torch.randint(midpoint, vocab_size, (batch_size, seq_len - split), generator=generator, device=device)
    input_ids = torch.cat([title_tokens, content_tokens], dim=1)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    return MSMBatch(input_ids=input_ids, field_ids=field_ids, attention_mask=attention_mask)


def sample_batch_summary(batch: MaskedMSMBatch, *, max_tokens: int = 16) -> dict[str, object]:
    """Return a small JSON-friendly batch summary for debugging."""

    limit = min(max_tokens, batch.input_ids.shape[1])
    return {
        "input_ids": batch.input_ids[0, :limit].detach().cpu().tolist(),
        "masked_field_ids": batch.field_ids[0, :limit].detach().cpu().tolist(),
        "original_field_ids": batch.original_field_ids[0, :limit].detach().cpu().tolist(),
        "labels": batch.labels[0, :limit].detach().cpu().tolist(),
        "mask_positions": batch.mask_positions[0, :limit].detach().cpu().tolist(),
        "total_masked": int(batch.mask_positions.sum().detach().cpu().item()),
        "masked_per_row": batch.mask_positions.sum(dim=1).detach().cpu().tolist(),
    }
