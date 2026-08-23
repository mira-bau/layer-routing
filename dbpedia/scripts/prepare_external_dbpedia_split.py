#!/usr/bin/env python3
"""Encode one local DBpedia split with an existing training tokenizer.

This is intended for held-out splits such as the original DBpedia benchmark
test CSV. It never retrains the tokenizer and records hashes needed by the
untouched-test analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from structformer.data.dbpedia import load_tokenizer, prepare_dbpedia_split  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--expected-records", type=int, default=None)
    args = parser.parse_args(argv)

    for path in (args.input_csv, args.tokenizer_json):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.max_length <= 0:
        raise ValueError("max-length must be positive")
    if args.expected_records is not None and args.expected_records <= 0:
        raise ValueError("expected-records must be positive")

    tokenizer = load_tokenizer(args.tokenizer_json)
    summary = prepare_dbpedia_split(
        args.input_csv,
        args.output_jsonl,
        tokenizer,
        max_length=args.max_length,
    )
    if args.expected_records is not None and summary["records"] != args.expected_records:
        raise ValueError(
            f"expected {args.expected_records} records, found {summary['records']}"
        )

    manifest = {
        "dataset": "DBpedia Ontology Classification",
        "split": args.split_name,
        "source_csv": str(args.input_csv),
        "source_csv_sha256": _sha256(args.input_csv),
        "tokenizer_json": str(args.tokenizer_json),
        "tokenizer_json_sha256": _sha256(args.tokenizer_json),
        "max_length": args.max_length,
        "records": summary["records"],
        "output_jsonl": str(args.output_jsonl),
        "output_jsonl_sha256": _sha256(args.output_jsonl),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"prepared-external-dbpedia split={args.split_name} "
        f"records={summary['records']} max_length={args.max_length}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
