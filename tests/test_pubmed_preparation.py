from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = REPO_ROOT / "pubmed/scripts/prepare_pubmed.py"
    spec = importlib.util.spec_from_file_location("prepare_pubmed", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_raw(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "###12345",
                "BACKGROUND\tignored",
                "METHODS\tFirst method sentence.",
                "METHODS\tSecond method sentence.",
                "RESULTS\tResult sentence.",
                "CONCLUSIONS\tConclusion sentence.",
                "",
                "###67890",
                "METHODS\tOther method.",
                "RESULTS\tOther result.",
                "CONCLUSIONS\tOther conclusion.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_release_preparation_matches_original_pubmed_metadata(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tokenizers")
    module = _load_module()
    train_path = tmp_path / "train.txt"
    val_path = tmp_path / "dev.txt"
    _write_raw(train_path)
    _write_raw(val_path)
    out_dir = tmp_path / "processed"

    assert (
        module.main(
            [
                "--train-txt",
                str(train_path),
                "--val-txt",
                str(val_path),
                "--out-dir",
                str(out_dir),
                "--vocab-size",
                "64",
                "--max-length",
                "32",
            ]
        )
        == 0
    )

    first = json.loads(
        (out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert first["row_id"] == "train_0"
    assert first["label"] == "12345"
    assert set(first["field_ids"]) == {3, 4, 5}

    manifest = json.loads(
        (out_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source"] == "PubMed_200k_RCT"
    assert manifest["fields"] == ["METHODS", "RESULTS", "CONCLUSIONS"]
    assert manifest["field_ids"] == {
        "METHODS": 3,
        "RESULTS": 4,
        "CONCLUSIONS": 5,
    }
    assert manifest["splits"]["train"]["records"] == 2
    assert "avg_length" in manifest["splits"]["train"]
