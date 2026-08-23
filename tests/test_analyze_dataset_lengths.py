from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_dataset_lengths import _read_dbpedia_dir


def test_reads_dbpedia_from_public_prepared_directory(tmp_path: Path) -> None:
    records = [
        {
            "row_id": "row_0",
            "input_ids": [2, 3, 4],
            "field_ids": [4, 4, 3],
            "attention_mask": [1, 1, 1],
        },
        {
            "row_id": "row_1",
            "input_ids": [2, 3],
            "field_ids": [4, 3],
            "attention_mask": [1, 1],
        },
    ]
    for split in ("train", "val"):
        (tmp_path / f"{split}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "max_length": 256,
                "splits": {"train": {"records": 2}, "val": {"records": 2}},
            }
        ),
        encoding="utf-8",
    )

    lengths, summaries, provenance = _read_dbpedia_dir(
        tmp_path, ["train", "val"]
    )

    assert lengths == {"train": [3, 2], "val": [3, 2]}
    assert {row["dataset"] for row in summaries} == {"DBpedia"}
    assert provenance["dataset"] == "DBpedia"
