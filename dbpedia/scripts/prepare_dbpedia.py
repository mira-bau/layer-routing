#!/usr/bin/env python3
"""Prepare local DBpedia CSV files for MSM.

Expected CSV columns:

  label,title,content

No dataset downloads are performed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from structformer.data.dbpedia import (  # noqa: E402
    iter_dbpedia_csv,
    prepare_dbpedia_split,
    save_tokenizer,
    train_dbpedia_tokenizer,
    write_field_vocab,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare local DBpedia CSV files for MSM.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=30_000)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args(argv)

    try:
        print(f"[prepare] training tokenizer from {args.train_csv}", flush=True)
        tokenizer = train_dbpedia_tokenizer(iter_dbpedia_csv(args.train_csv), vocab_size=args.vocab_size)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        save_tokenizer(tokenizer, args.out_dir / "tokenizer.json")
        write_field_vocab(args.out_dir / "field_vocab.json")

        splits = {}
        print(f"[prepare] encoding split=train source={args.train_csv}", flush=True)
        splits["train"] = prepare_dbpedia_split(args.train_csv, args.out_dir / "train.jsonl", tokenizer, max_length=args.max_length)
        print(f"[prepare] split=train records={splits['train']['records']} max_len={splits['train']['max_length']}", flush=True)
        if args.val_csv is not None:
            print(f"[prepare] encoding split=val source={args.val_csv}", flush=True)
            splits["val"] = prepare_dbpedia_split(args.val_csv, args.out_dir / "val.jsonl", tokenizer, max_length=args.max_length)
            print(f"[prepare] split=val records={splits['val']['records']} max_len={splits['val']['max_length']}", flush=True)
        if args.test_csv is not None:
            print(f"[prepare] encoding split=test source={args.test_csv}", flush=True)
            splits["test"] = prepare_dbpedia_split(args.test_csv, args.out_dir / "test.jsonl", tokenizer, max_length=args.max_length)
            print(f"[prepare] split=test records={splits['test']['records']} max_len={splits['test']['max_length']}", flush=True)

        manifest = {
            "task": "dbpedia_msm",
            "vocab_size": args.vocab_size,
            "max_length": args.max_length,
            "tokenizer_source_split": "train",
            "splits": splits,
        }
        (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"DBpedia preparation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Prepared DBpedia MSM artifacts in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
