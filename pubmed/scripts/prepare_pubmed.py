#!/usr/bin/env python3
"""Prepare PubMed 200k RCT text files for three-field MSM.

Input is the standard PubMed-RCT format, where each abstract begins with a
``###<pmid>`` line followed by ``LABEL<TAB>sentence`` lines. Only abstracts that
contain all three of METHODS, RESULTS, and CONCLUSIONS are kept, and only those
three fields are used, linearized in that narrative order. The output matches
the JSONL/tokenizer format the MSM trainer expects, with field ids:

    <PAD>=0  [NONE]=1  <UNK>=2  methods=3  results=4  conclusions=5  [FIELD_MASK]=6

so it lines up with the PubMed configs (field_vocab_size=7, mask_field_id=6).
No data is downloaded; raw files are provided by path.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

PAD_FIELD, NONE_FIELD, UNK_FIELD = "<PAD>", "[NONE]", "<UNK>"
FIELD_START, FIELD_MASK = "[FIELD_START]", "[FIELD_MASK]"
DATA_FIELDS = ("methods", "results", "conclusions")
FIELD_VOCAB = {PAD_FIELD: 0, NONE_FIELD: 1, UNK_FIELD: 2,
               "methods": 3, "results": 4, "conclusions": 5}
FIELD_MASK_ID = 6


def iter_abstracts(path: Path) -> Iterable[dict[str, str]]:
    """Yield PMID and text fields for abstracts holding all three data fields."""
    current_pmid: str | None = None
    current: "OrderedDict[str, list[str]]" = OrderedDict()

    def flush() -> Iterable[dict[str, str]]:
        if current_pmid and all(f in current for f in DATA_FIELDS):
            yield {
                "pmid": current_pmid,
                **{f: " ".join(current[f]) for f in DATA_FIELDS},
            }

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("###"):
                yield from flush()
                current_pmid = line[3:]
                current = OrderedDict()
                continue
            label, _, sentence = line.partition("\t")
            field = label.strip().lower()
            if field in DATA_FIELDS and sentence.strip():
                current.setdefault(field, []).append(sentence.strip())
        yield from flush()


def train_tokenizer(texts: Iterable[str], *, vocab_size: int, min_frequency: int = 1):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token=UNK_FIELD))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=min_frequency,
        special_tokens=[PAD_FIELD, UNK_FIELD, FIELD_START], show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def encode(
    record: dict[str, str],
    tokenizer,
    *,
    row_id: str,
    max_length: int,
) -> dict[str, Any]:
    start_id = tokenizer.token_to_id(FIELD_START)
    if start_id is None:
        raise ValueError(f"Tokenizer is missing required special token {FIELD_START}")
    input_ids: list[int] = []
    field_ids: list[int] = []
    tokens: list[str] = []
    for field in DATA_FIELDS:
        fid = FIELD_VOCAB[field]
        input_ids.append(int(start_id)); field_ids.append(fid); tokens.append(FIELD_START)
        enc = tokenizer.encode(record[field], add_special_tokens=False)
        input_ids.extend(int(i) for i in enc.ids)
        field_ids.extend([fid] * len(enc.ids))
        tokens.extend(enc.tokens)
    input_ids, field_ids, tokens = input_ids[:max_length], field_ids[:max_length], tokens[:max_length]
    return {"row_id": row_id, "label": record["pmid"], "input_ids": input_ids,
            "field_ids": field_ids, "attention_mask": [1] * len(input_ids), "tokens": tokens}


def write_split(
    txt_path: Path,
    out_jsonl: Path,
    tokenizer,
    *,
    split_name: str,
    max_length: int,
) -> dict[str, Any]:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    lengths: list[int] = []
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for idx, record in enumerate(iter_abstracts(txt_path)):
            enc = encode(
                record,
                tokenizer,
                row_id=f"{split_name}_{idx}",
                max_length=max_length,
            )
            handle.write(json.dumps(enc, sort_keys=True) + "\n")
            lengths.append(len(enc["input_ids"]))
    return {
        "output": str(out_jsonl),
        "records": len(lengths),
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "avg_length": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Prepare PubMed 200k RCT for three-field MSM.")
    p.add_argument("--train-txt", type=Path, required=True)
    p.add_argument("--val-txt", type=Path, default=None)
    p.add_argument("--test-txt", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--vocab-size", type=int, default=30_000)
    p.add_argument("--max-length", type=int, default=512)
    args = p.parse_args(argv)

    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[prepare] training tokenizer from {args.train_txt}", flush=True)
        texts = (record[f] for record in iter_abstracts(args.train_txt) for f in DATA_FIELDS)
        tokenizer = train_tokenizer(texts, vocab_size=args.vocab_size)
        tokenizer.save(str(args.out_dir / "tokenizer.json"))

        (args.out_dir / "field_vocab.json").write_text(json.dumps({
            "field_vocab": FIELD_VOCAB,
            "mask_field": {"name": FIELD_MASK, "id": FIELD_MASK_ID},
            "msm_target_ids": [FIELD_VOCAB[f] for f in DATA_FIELDS],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        splits = {}
        for name, path in (("train", args.train_txt), ("val", args.val_txt), ("test", args.test_txt)):
            if path is None:
                continue
            print(f"[prepare] encoding split={name} source={path}", flush=True)
            splits[name] = write_split(
                path,
                args.out_dir / f"{name}.jsonl",
                tokenizer,
                split_name=name,
                max_length=args.max_length,
            )
            print(f"[prepare] split={name} records={splits[name]['records']} max_len={splits[name]['max_length']}", flush=True)

        (args.out_dir / "manifest.json").write_text(json.dumps({
            "task": "pubmed_msm",
            "source": "PubMed_200k_RCT",
            "fields": [field.upper() for field in DATA_FIELDS],
            "field_ids": {
                field.upper(): FIELD_VOCAB[field] for field in DATA_FIELDS
            },
            "vocab_size": args.vocab_size,
            "max_length": args.max_length,
            "tokenizer_source_split": "train", "splits": splits,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"PubMed preparation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Prepared PubMed MSM artifacts in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
