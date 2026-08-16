"""DBpedia MSM preprocessing helpers.

This module turns local `label,title,content` CSV rows into aligned token and
field ID sequences. It intentionally does not download data.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from typing import Any


PAD_FIELD = "<PAD>"
NONE_FIELD = "[NONE]"
UNK_FIELD = "<UNK>"
CONTENT_FIELD = "content"
TITLE_FIELD = "title"
FIELD_MASK = "[FIELD_MASK]"
FIELD_START = "[FIELD_START]"

FIELD_VOCAB: dict[str, int] = {
    PAD_FIELD: 0,
    NONE_FIELD: 1,
    UNK_FIELD: 2,
    CONTENT_FIELD: 3,
    TITLE_FIELD: 4,
}
FIELD_MASK_ID = 5
DATA_FIELDS = (TITLE_FIELD, CONTENT_FIELD)
CSV_COLUMNS = ("label", TITLE_FIELD, CONTENT_FIELD)


@dataclass(frozen=True)
class DBpediaRow:
    row_id: str
    label: str
    title: str
    content: str


@dataclass(frozen=True)
class EncodedDBpediaRecord:
    row_id: str
    label: str
    input_ids: list[int]
    field_ids: list[int]
    attention_mask: list[int]
    tokens: list[str]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def read_dbpedia_csv(path: str | Path) -> list[DBpediaRow]:
    """Read a local DBpedia CSV with columns `label,title,content`."""

    return list(iter_dbpedia_csv(path))


def iter_dbpedia_csv(path: str | Path) -> Iterable[DBpediaRow]:
    """Stream headered or original headerless DBpedia CSV rows."""

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        first = next(reader, None)
        if first is None:
            raise ValueError("DBpedia CSV is empty")
        normalized_first = tuple(value.strip().lower() for value in first)
        if normalized_first == CSV_COLUMNS:
            data_rows = reader
        elif any(value in CSV_COLUMNS for value in normalized_first):
            missing = [column for column in CSV_COLUMNS if column not in normalized_first]
            raise ValueError(f"DBpedia CSV is missing required columns: {missing}")
        else:
            data_rows = chain([first], reader)

        record_index = 0
        for line_index, values in enumerate(data_rows, start=1):
            if not values or not any(value.strip() for value in values):
                continue
            if len(values) != len(CSV_COLUMNS):
                raise ValueError(
                    f"DBpedia CSV row {line_index} has {len(values)} columns; "
                    f"expected {len(CSV_COLUMNS)} in label,title,content order"
                )
            row = dict(zip(CSV_COLUMNS, values))
            yield DBpediaRow(
                row_id=f"row_{record_index}",
                label=(row.get("label") or "").strip(),
                title=(row.get(TITLE_FIELD) or "").strip(),
                content=(row.get(CONTENT_FIELD) or "").strip(),
            )
            record_index += 1


def train_dbpedia_tokenizer(
    train_rows: Iterable[DBpediaRow],
    *,
    vocab_size: int,
    min_frequency: int = 1,
):
    """Train a BPE tokenizer on DBpedia train rows only."""

    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token=UNK_FIELD))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<PAD>", UNK_FIELD, FIELD_START],
        show_progress=False,
    )
    tokenizer.train_from_iterator(_training_texts(train_rows), trainer=trainer)
    return tokenizer


def load_tokenizer(path: str | Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def save_tokenizer(tokenizer, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))


def encode_dbpedia_row(row: DBpediaRow, tokenizer, *, max_length: int) -> EncodedDBpediaRecord:
    """Encode one row with title -> content linearization and aligned fields."""

    input_ids: list[int] = []
    field_ids: list[int] = []
    tokens: list[str] = []

    for field_name in DATA_FIELDS:
        field_id = FIELD_VOCAB[field_name]
        start_id = tokenizer.token_to_id(FIELD_START)
        if start_id is None:
            raise ValueError(f"Tokenizer does not contain required special token {FIELD_START}")

        input_ids.append(int(start_id))
        field_ids.append(field_id)
        tokens.append(FIELD_START)

        text = getattr(row, field_name)
        encoded = tokenizer.encode(text, add_special_tokens=False)
        input_ids.extend(int(token_id) for token_id in encoded.ids)
        field_ids.extend([field_id] * len(encoded.ids))
        tokens.extend(encoded.tokens)

    input_ids = input_ids[:max_length]
    field_ids = field_ids[:max_length]
    tokens = tokens[:max_length]
    attention_mask = [1] * len(input_ids)

    return EncodedDBpediaRecord(
        row_id=row.row_id,
        label=row.label,
        input_ids=input_ids,
        field_ids=field_ids,
        attention_mask=attention_mask,
        tokens=tokens,
    )


def write_jsonl(records: Iterable[EncodedDBpediaRecord], path: str | Path) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json(), sort_keys=True) + "\n")
            count += 1
    return count


def prepare_dbpedia_split(
    csv_path: str | Path,
    output_jsonl: str | Path,
    tokenizer,
    *,
    max_length: int,
) -> dict[str, Any]:
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    min_length: int | None = None
    max_observed_length = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in iter_dbpedia_csv(csv_path):
            record = encode_dbpedia_row(row, tokenizer, max_length=max_length)
            length = len(record.input_ids)
            min_length = length if min_length is None else min(min_length, length)
            max_observed_length = max(max_observed_length, length)
            handle.write(json.dumps(record.to_json(), sort_keys=True) + "\n")
            count += 1
    return {
        "source": str(csv_path),
        "output": str(output_jsonl),
        "records": count,
        "min_length": min_length or 0,
        "max_length": max_observed_length,
    }


def write_field_vocab(path: str | Path) -> None:
    payload = {
        "field_vocab": FIELD_VOCAB,
        "mask_field": {"name": FIELD_MASK, "id": FIELD_MASK_ID},
        "msm_target_ids": [FIELD_VOCAB[CONTENT_FIELD], FIELD_VOCAB[TITLE_FIELD]],
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _training_texts(rows: Iterable[DBpediaRow]) -> Iterable[str]:
    for row in rows:
        yield row.title
        yield row.content
