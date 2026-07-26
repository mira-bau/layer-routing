#!/usr/bin/env python3
"""Create the DBpedia train/validation split used by the paper.

The input is the original 560,000-example DBpedia training CSV with columns
in this order (a header row is optional):

  label,title,content

No dataset downloads are performed. The benchmark test split is not read or
modified by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


CSV_COLUMNS = ("label", "title", "content")
DEFAULT_VALIDATION_SIZE = 70_000
DEFAULT_SPLIT_SEED = 42


def split_dbpedia_csv(
    source_csv: str | Path,
    out_dir: str | Path,
    *,
    validation_size: int = DEFAULT_VALIDATION_SIZE,
    split_seed: int = DEFAULT_SPLIT_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Split a local DBpedia training CSV deterministically.

    The ordering matches an integer-sized ``train_test_split`` with
    ``random_state=split_seed`` and ``shuffle=True``: validation rows are the
    first ``validation_size`` entries of the permutation, and training rows are
    the remainder.
    """

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "NumPy is required for DBpedia split generation. "
            "Install the project runtime dependencies manually."
        ) from exc

    source_path = Path(source_csv)
    output_dir = Path(out_dir)
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    manifest_path = output_dir / "split_manifest.json"

    if not source_path.is_file():
        raise FileNotFoundError(f"DBpedia source CSV does not exist: {source_path}")
    if not 0 <= split_seed <= 2**32 - 1:
        raise ValueError("split_seed must be between 0 and 4294967295")
    if validation_size <= 0:
        raise ValueError("validation_size must be greater than zero")

    source_resolved = source_path.resolve()
    if source_resolved in {train_path.resolve(), val_path.resolve()}:
        raise ValueError("out_dir must not overwrite the source DBpedia CSV")

    outputs = (train_path, val_path, manifest_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output files already exist: {names}. Pass --overwrite to replace them."
        )

    rows = _read_rows(source_path)
    source_size = len(rows)
    if validation_size >= source_size:
        raise ValueError(
            "validation_size must be smaller than the number of source records "
            f"({source_size})"
        )

    permutation = np.random.RandomState(split_seed).permutation(source_size)
    val_indices = permutation[:validation_size]
    train_indices = permutation[validation_size:]

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(train_path, rows, train_indices)
    _write_rows(val_path, rows, val_indices)

    manifest: dict[str, Any] = {
        "dataset": "DBpedia Ontology Classification",
        "source": {
            "csv": str(source_path),
            "records": source_size,
            "sha256": _sha256(source_path),
        },
        "split": {
            "method": "numpy.random.RandomState(seed).permutation",
            "seed": split_seed,
            "shuffle": True,
            "validation_size": validation_size,
            "benchmark_test_split_used": False,
        },
        "outputs": {
            "train": {
                "csv": str(train_path),
                "records": len(train_indices),
                "sha256": _sha256(train_path),
            },
            "validation": {
                "csv": str(val_path),
                "records": len(val_indices),
                "sha256": _sha256(val_path),
            },
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError("DBpedia source CSV is empty")

    first = tuple(value.strip().lower() for value in raw_rows[0][:3])
    data_rows = raw_rows[1:] if first == CSV_COLUMNS else raw_rows
    rows: list[dict[str, str]] = []
    for line_number, row in enumerate(
        data_rows,
        start=2 if first == CSV_COLUMNS else 1,
    ):
        if len(row) != len(CSV_COLUMNS):
            raise ValueError(
                f"DBpedia CSV row {line_number} has {len(row)} columns; "
                f"expected {len(CSV_COLUMNS)} in label,title,content order"
            )
        rows.append(dict(zip(CSV_COLUMNS, row)))
    return rows


def _write_rows(
    path: Path,
    rows: list[dict[str, str]],
    indices: Any,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for index in indices:
            writer.writerow(rows[int(index)])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create the deterministic DBpedia split used by the paper."
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        required=True,
        help=(
            "Original DBpedia training CSV with 560,000 examples. "
            "A label,title,content header is optional."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--validation-size",
        type=int,
        default=DEFAULT_VALIDATION_SIZE,
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace train.csv, val.csv, and split_manifest.json in out-dir.",
    )
    args = parser.parse_args(argv)

    print(
        "[split] "
        f"source={args.source_csv} validation_size={args.validation_size} "
        f"seed={args.split_seed}",
        flush=True,
    )
    try:
        manifest = split_dbpedia_csv(
            args.source_csv,
            args.out_dir,
            validation_size=args.validation_size,
            split_seed=args.split_seed,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"DBpedia split generation failed: {exc}", file=sys.stderr)
        return 1

    train_records = manifest["outputs"]["train"]["records"]
    val_records = manifest["outputs"]["validation"]["records"]
    print(
        f"[split] train_records={train_records} validation_records={val_records}",
        flush=True,
    )
    print(f"Created DBpedia split in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
