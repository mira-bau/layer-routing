from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "dbpedia"
    / "scripts"
    / "prepare_external_dbpedia_split.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("prepare_external_dbpedia_split", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_external_split_reuses_tokenizer_and_writes_analysis_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    source_csv = tmp_path / "test.csv"
    tokenizer_json = tmp_path / "tokenizer.json"
    output_jsonl = tmp_path / "prepared" / "test.jsonl"
    manifest_path = tmp_path / "prepared" / "test_manifest.json"
    source_csv.write_text("1,title,content\n2,other,record\n", encoding="utf-8")
    tokenizer_json.write_text('{"tokenizer":"retained"}\n', encoding="utf-8")
    retained_tokenizer = object()

    def fake_load_tokenizer(path: Path):
        assert path == tokenizer_json
        return retained_tokenizer

    def fake_prepare(source: Path, output: Path, tokenizer, *, max_length: int):
        assert source == source_csv
        assert tokenizer is retained_tokenizer
        assert max_length == 256
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{"row_id":"test_0"}\n{"row_id":"test_1"}\n', encoding="utf-8")
        return {"records": 2}

    monkeypatch.setattr(module, "load_tokenizer", fake_load_tokenizer)
    monkeypatch.setattr(module, "prepare_dbpedia_split", fake_prepare)

    assert (
        module.main(
            [
                "--input-csv",
                str(source_csv),
                "--tokenizer-json",
                str(tokenizer_json),
                "--output-jsonl",
                str(output_jsonl),
                "--manifest",
                str(manifest_path),
                "--expected-records",
                "2",
            ]
        )
        == 0
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest == {
        "dataset": "DBpedia Ontology Classification",
        "split": "test",
        "source_csv": str(source_csv),
        "source_csv_sha256": _sha256(source_csv),
        "tokenizer_json": str(tokenizer_json),
        "tokenizer_json_sha256": _sha256(tokenizer_json),
        "max_length": 256,
        "records": 2,
        "output_jsonl": str(output_jsonl),
        "output_jsonl_sha256": _sha256(output_jsonl),
    }


def test_external_split_rejects_unexpected_record_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    source_csv = tmp_path / "test.csv"
    tokenizer_json = tmp_path / "tokenizer.json"
    output_jsonl = tmp_path / "test.jsonl"
    manifest_path = tmp_path / "test_manifest.json"
    source_csv.write_text("1,title,content\n", encoding="utf-8")
    tokenizer_json.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(module, "load_tokenizer", lambda _: object())

    def fake_prepare(*_args, **_kwargs):
        output_jsonl.write_text("{}\n", encoding="utf-8")
        return {"records": 1}

    monkeypatch.setattr(module, "prepare_dbpedia_split", fake_prepare)

    with pytest.raises(ValueError, match="expected 70000 records, found 1"):
        module.main(
            [
                "--input-csv",
                str(source_csv),
                "--tokenizer-json",
                str(tokenizer_json),
                "--output-jsonl",
                str(output_jsonl),
                "--manifest",
                str(manifest_path),
                "--expected-records",
                "70000",
            ]
        )
    assert not manifest_path.exists()
