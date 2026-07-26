"""Prepared JSONL datasets for MSM training."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from structformer.tasks.msm import MSMBatch


@dataclass(frozen=True)
class PreparedMSMRecord:
    row_id: str
    input_ids: list[int]
    field_ids: list[int]
    attention_mask: list[int]
    label: str | None = None
    tokens: list[str] | None = None


@dataclass(frozen=True)
class PreparedMSMCollatedBatch:
    batch: MSMBatch
    row_ids: list[str]
    tokens: list[list[str] | None]


class PreparedMSMJsonlDataset(Dataset[PreparedMSMRecord]):
    """Read prepared MSM JSONL records into memory.

    This is deliberately simple for the first training path. A sharded tensor
    format can be added later for large-scale throughput.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_records: int | None = None,
        sample_size: int | None = None,
        sample_seed: int | None = None,
    ) -> None:
        self.path = Path(path)
        if max_records is not None and sample_size is not None:
            raise ValueError("max_records and sample_size are mutually exclusive")

        self.sample_size = _positive_int("sample_size", sample_size)
        self.sample_seed: int | None = None
        self.sample_indices_hash: str | None = None
        self.sample_indices_preview: list[int] = []

        if self.sample_size is not None:
            seed = 0 if sample_seed is None else int(sample_seed)
            self.records, indices, self.source_records = _read_sampled_records(
                self.path,
                sample_size=self.sample_size,
                sample_seed=seed,
            )
            self.sample_seed = seed
            self.sample_indices_hash = _hash_indices(indices)
            self.sample_indices_preview = indices[:10]
        else:
            max_records = _positive_int("max_records", max_records)
            self.records = _read_records(self.path, max_records=max_records)
            self.source_records = len(self.records)

        if not self.records:
            raise ValueError(f"No records found in {self.path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> PreparedMSMRecord:
        return self.records[index]


def collate_prepared_msm(
    records: list[PreparedMSMRecord],
    *,
    pad_token_id: int = 0,
    pad_field_id: int = 0,
    max_length: int | None = None,
) -> PreparedMSMCollatedBatch:
    if not records:
        raise ValueError("Cannot collate an empty batch")

    batch_max = max(len(record.input_ids) for record in records)
    target_len = min(batch_max, max_length) if max_length is not None else batch_max
    input_ids = torch.full((len(records), target_len), pad_token_id, dtype=torch.long)
    field_ids = torch.full((len(records), target_len), pad_field_id, dtype=torch.long)
    attention_mask = torch.zeros((len(records), target_len), dtype=torch.bool)

    row_ids: list[str] = []
    tokens: list[list[str] | None] = []
    for row, record in enumerate(records):
        length = min(len(record.input_ids), target_len)
        input_ids[row, :length] = torch.tensor(record.input_ids[:length], dtype=torch.long)
        field_ids[row, :length] = torch.tensor(record.field_ids[:length], dtype=torch.long)
        attention_mask[row, :length] = torch.tensor(record.attention_mask[:length], dtype=torch.bool)
        row_ids.append(record.row_id)
        tokens.append(record.tokens[:length] if record.tokens is not None else None)

    return PreparedMSMCollatedBatch(
        batch=MSMBatch(input_ids=input_ids, field_ids=field_ids, attention_mask=attention_mask),
        row_ids=row_ids,
        tokens=tokens,
    )


def _read_records(path: Path, *, max_records: int | None) -> list[PreparedMSMRecord]:
    records: list[PreparedMSMRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_records is not None and len(records) >= max_records:
                break
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(_record_from_json(payload, path=path, line_number=line_number))
    return records


def _read_sampled_records(
    path: Path,
    *,
    sample_size: int,
    sample_seed: int,
) -> tuple[list[PreparedMSMRecord], list[int], int]:
    rng = random.Random(sample_seed)
    reservoir: list[tuple[int, PreparedMSMRecord]] = []
    source_records = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            record = _record_from_json(payload, path=path, line_number=line_number)
            record_index = source_records
            source_records += 1

            if len(reservoir) < sample_size:
                reservoir.append((record_index, record))
                continue

            replace_at = rng.randrange(source_records)
            if replace_at < sample_size:
                reservoir[replace_at] = (record_index, record)

    if source_records < sample_size:
        raise ValueError(f"sample_size={sample_size} exceeds {source_records} records in {path}")

    reservoir.sort(key=lambda item: item[0])
    indices = [index for index, _ in reservoir]
    records = [record for _, record in reservoir]
    return records, indices, source_records


def _positive_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _hash_indices(indices: list[int]) -> str:
    payload = json.dumps(indices, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _record_from_json(payload: dict[str, Any], *, path: Path, line_number: int) -> PreparedMSMRecord:
    required = ["row_id", "input_ids", "field_ids", "attention_mask"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{path}:{line_number} missing required keys: {missing}")

    input_ids = [int(value) for value in payload["input_ids"]]
    field_ids = [int(value) for value in payload["field_ids"]]
    attention_mask = [int(value) for value in payload["attention_mask"]]
    if not (len(input_ids) == len(field_ids) == len(attention_mask)):
        raise ValueError(f"{path}:{line_number} has misaligned input_ids/field_ids/attention_mask")

    return PreparedMSMRecord(
        row_id=str(payload["row_id"]),
        label=str(payload["label"]) if payload.get("label") is not None else None,
        input_ids=input_ids,
        field_ids=field_ids,
        attention_mask=attention_mask,
        tokens=[str(token) for token in payload.get("tokens", [])] if "tokens" in payload else None,
    )
