from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_split_module():
    path = REPO_ROOT / "dbpedia/scripts/split_dbpedia.py"
    spec = importlib.util.spec_from_file_location("split_dbpedia", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_source(path: Path, count: int = 12) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "title", "content"],
        )
        writer.writeheader()
        for index in range(count):
            writer.writerow(
                {
                    "label": str(index % 3),
                    "title": f"title {index}",
                    "content": f"content {index}",
                }
            )


def _read_titles(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [row["title"] for row in csv.DictReader(handle)]


def test_split_is_deterministic_and_records_manifest(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    module = _load_split_module()
    source = tmp_path / "source.csv"
    _write_source(source)

    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = module.split_dbpedia_csv(
        source,
        first,
        validation_size=3,
        split_seed=42,
    )
    module.split_dbpedia_csv(
        source,
        second,
        validation_size=3,
        split_seed=42,
    )

    expected = np.random.RandomState(42).permutation(12)
    assert _read_titles(first / "val.csv") == [
        f"title {index}" for index in expected[:3]
    ]
    assert _read_titles(first / "train.csv") == [
        f"title {index}" for index in expected[3:]
    ]
    assert (first / "train.csv").read_bytes() == (second / "train.csv").read_bytes()
    assert (first / "val.csv").read_bytes() == (second / "val.csv").read_bytes()
    assert manifest["split"]["benchmark_test_split_used"] is False
    assert manifest["outputs"]["train"]["records"] == 9
    assert manifest["outputs"]["validation"]["records"] == 3


def test_split_refuses_to_overwrite(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    module = _load_split_module()
    source = tmp_path / "source.csv"
    _write_source(source)
    output = tmp_path / "split"
    module.split_dbpedia_csv(source, output, validation_size=3)
    with pytest.raises(FileExistsError):
        module.split_dbpedia_csv(source, output, validation_size=3)


def test_split_accepts_original_headerless_csv(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    module = _load_split_module()
    source = tmp_path / "source.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for index in range(6):
            writer.writerow([index % 2, f"title {index}", f"content {index}"])

    output = tmp_path / "split"
    manifest = module.split_dbpedia_csv(source, output, validation_size=2)
    assert manifest["source"]["records"] == 6
    assert len(_read_titles(output / "train.csv")) == 4
    assert len(_read_titles(output / "val.csv")) == 2
