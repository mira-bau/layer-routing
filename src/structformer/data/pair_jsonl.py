"""Prepared pair JSONL datasets for sequence classification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PreparedPairRecord:
    pair_id: str
    a_row_id: str
    b_row_id: str
    label: int
    input_ids: list[int]
    field_ids: list[int]
    attention_mask: list[int]
    tokens: list[str] | None = None


@dataclass(frozen=True)
class PairBatch:
    input_ids: torch.Tensor
    field_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


@dataclass(frozen=True)
class PreparedPairCollatedBatch:
    batch: PairBatch
    pair_ids: list[str]
    row_pairs: list[tuple[str, str]]
    tokens: list[list[str] | None]


class PreparedPairJsonlDataset(Dataset[PreparedPairRecord]):
    """Read prepared pair JSONL records into memory."""

    def __init__(self, path: str | Path, *, max_records: int | None = None) -> None:
        self.path = Path(path)
        self.records = _read_records(self.path, max_records=max_records)
        if not self.records:
            raise ValueError(f"No records found in {self.path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> PreparedPairRecord:
        return self.records[index]


def collate_prepared_pairs(
    records: list[PreparedPairRecord],
    *,
    pad_token_id: int = 0,
    pad_field_id: int = 0,
    max_length: int | None = None,
) -> PreparedPairCollatedBatch:
    if not records:
        raise ValueError("Cannot collate an empty batch")

    batch_max = max(len(record.input_ids) for record in records)
    target_len = min(batch_max, max_length) if max_length is not None else batch_max
    input_ids = torch.full((len(records), target_len), pad_token_id, dtype=torch.long)
    field_ids = torch.full((len(records), target_len), pad_field_id, dtype=torch.long)
    attention_mask = torch.zeros((len(records), target_len), dtype=torch.bool)
    labels = torch.empty(len(records), dtype=torch.long)

    pair_ids: list[str] = []
    row_pairs: list[tuple[str, str]] = []
    tokens: list[list[str] | None] = []
    for row, record in enumerate(records):
        length = min(len(record.input_ids), target_len)
        input_ids[row, :length] = torch.tensor(record.input_ids[:length], dtype=torch.long)
        field_ids[row, :length] = torch.tensor(record.field_ids[:length], dtype=torch.long)
        attention_mask[row, :length] = torch.tensor(record.attention_mask[:length], dtype=torch.bool)
        labels[row] = int(record.label)
        pair_ids.append(record.pair_id)
        row_pairs.append((record.a_row_id, record.b_row_id))
        tokens.append(record.tokens[:length] if record.tokens is not None else None)

    return PreparedPairCollatedBatch(
        batch=PairBatch(input_ids=input_ids, field_ids=field_ids, attention_mask=attention_mask, labels=labels),
        pair_ids=pair_ids,
        row_pairs=row_pairs,
        tokens=tokens,
    )


def _read_records(path: Path, *, max_records: int | None) -> list[PreparedPairRecord]:
    records: list[PreparedPairRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_records is not None and len(records) >= max_records:
                break
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(_record_from_json(payload, path=path, line_number=line_number))
    return records


def _record_from_json(payload: dict[str, Any], *, path: Path, line_number: int) -> PreparedPairRecord:
    required = ["pair_id", "a_row_id", "b_row_id", "label", "input_ids", "field_ids", "attention_mask"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{path}:{line_number} missing required keys: {missing}")

    input_ids = [int(value) for value in payload["input_ids"]]
    field_ids = [int(value) for value in payload["field_ids"]]
    attention_mask = [int(value) for value in payload["attention_mask"]]
    if not (len(input_ids) == len(field_ids) == len(attention_mask)):
        raise ValueError(f"{path}:{line_number} has misaligned input_ids/field_ids/attention_mask")

    label = int(payload["label"])
    if label not in {0, 1}:
        raise ValueError(f"{path}:{line_number} label must be 0 or 1")

    return PreparedPairRecord(
        pair_id=str(payload["pair_id"]),
        a_row_id=str(payload["a_row_id"]),
        b_row_id=str(payload["b_row_id"]),
        label=label,
        input_ids=input_ids,
        field_ids=field_ids,
        attention_mask=attention_mask,
        tokens=[str(token) for token in payload.get("tokens", [])] if "tokens" in payload else None,
    )

