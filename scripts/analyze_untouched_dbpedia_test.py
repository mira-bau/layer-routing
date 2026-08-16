#!/usr/bin/env python3
"""Confirm seed-1001 DBpedia routing on untouched benchmark test records."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_opportunity_normalized_sfm import (
    adjusted_sfm,
    opportunity_reference,
)
from scripts.analyze_paired_sfm_length import (
    DatasetSpec,
    _evaluate_model,
    _field_ids,
    _load_checkpoint,
    _paired_stats,
    _rebuild_model,
    _select_records,
    _sha256,
    _verify_checkpoint,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_rows(
    records: list[dict],
    baseline: dict[int, list[float]],
    saab: dict[int, list[float]],
    named_fields: list[int],
    max_length: int,
) -> list[dict]:
    rows = []
    for record in records:
        index = int(record["source_index"])
        opportunity = opportunity_reference(
            record["field_ids"],
            record["attention_mask"],
            set(named_fields),
            max_length,
        )
        row = {
            "dataset": "DBpedia-test",
            "source_index": index,
            "row_id": record.get("row_id"),
            "evaluation_token_length": min(
                int(sum(record["attention_mask"])), max_length
            ),
            "evaluation_max_length": max_length,
            "opportunity_reference": opportunity,
        }
        for layer in range(4):
            baseline_value = baseline[index][layer]
            saab_value = saab[index][layer]
            baseline_adjusted = adjusted_sfm(baseline_value, opportunity)
            saab_adjusted = adjusted_sfm(saab_value, opportunity)
            row[f"baseline_l{layer}"] = baseline_value
            row[f"saab_l{layer}"] = saab_value
            row[f"delta_l{layer}"] = saab_value - baseline_value
            row[f"baseline_adjusted_l{layer}"] = baseline_adjusted
            row[f"saab_adjusted_l{layer}"] = saab_adjusted
            row[f"delta_adjusted_l{layer}"] = (
                None
                if baseline_adjusted is None or saab_adjusted is None
                else saab_adjusted - baseline_adjusted
            )
        rows.append(row)
    return rows


def _adjusted_stats_rows(rows: list[dict]) -> list[dict]:
    adjusted = []
    for row in rows:
        if row["delta_adjusted_l0"] is None:
            continue
        result = dict(row)
        for layer in range(4):
            result[f"baseline_l{layer}"] = row[f"baseline_adjusted_l{layer}"]
            result[f"saab_l{layer}"] = row[f"saab_adjusted_l{layer}"]
            result[f"delta_l{layer}"] = row[f"delta_adjusted_l{layer}"]
        adjusted.append(result)
    return adjusted


def run(args) -> Path:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Select a Colab GPU runtime.")
    device = torch.device("cuda")
    spec = DatasetSpec(
        "DBpedia-test",
        args.baseline_checkpoint,
        args.saab_checkpoint,
        args.test_jsonl,
        args.field_vocab_json,
        500,
        256,
        5,
        6,
    )
    for path in (
        spec.baseline_checkpoint,
        spec.saab_checkpoint,
        spec.validation_jsonl,
        spec.field_vocab_json,
        args.test_manifest,
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(args.test_manifest.read_text(encoding="utf-8"))
    if manifest.get("split") != "test" or manifest.get("records") != 70_000:
        raise ValueError("test manifest does not describe the 70,000-record split")
    if manifest.get("output_jsonl_sha256") != _sha256(args.test_jsonl):
        raise ValueError("test JSONL hash does not match its preparation manifest")

    baseline_checkpoint = _load_checkpoint(spec.baseline_checkpoint)
    saab_checkpoint = _load_checkpoint(spec.saab_checkpoint)
    _verify_checkpoint(baseline_checkpoint, spec, "baseline")
    _verify_checkpoint(saab_checkpoint, spec, "saab")

    selected, _, sample_indices, source_records = _select_records(
        spec.validation_jsonl,
        primary_size=1,
        length_sample_size=args.sample_size,
        sample_seed=args.analysis_seed,
    )
    records = [
        record for record in selected if int(record["source_index"]) in sample_indices
    ]
    if len(records) != args.sample_size:
        raise ValueError(f"expected {args.sample_size} selected records, found {len(records)}")
    named_fields = _field_ids(
        spec.field_vocab_json,
        int(baseline_checkpoint["config"]["data"]["mask_field_id"]),
    )

    baseline_model = _rebuild_model(baseline_checkpoint, device)
    baseline = _evaluate_model(
        baseline_model,
        records,
        named_field_ids=named_fields,
        batch_size=args.eval_batch_size,
        device=device,
        dataset="DBpedia-test",
        variant="baseline",
        evaluation_max_length=256,
    )
    del baseline_model, baseline_checkpoint
    torch.cuda.empty_cache()

    saab_model = _rebuild_model(saab_checkpoint, device)
    saab = _evaluate_model(
        saab_model,
        records,
        named_field_ids=named_fields,
        batch_size=args.eval_batch_size,
        device=device,
        dataset="DBpedia-test",
        variant="saab",
        evaluation_max_length=256,
    )
    del saab_model, saab_checkpoint
    torch.cuda.empty_cache()

    rows = _make_rows(records, baseline, saab, named_fields, 256)
    adjusted_rows = _adjusted_stats_rows(rows)
    raw_stats = _paired_stats(rows, args, "DBpedia-test-raw")
    adjusted_stats = _paired_stats(
        adjusted_rows, args, "DBpedia-test-opportunity-adjusted"
    )
    for row in adjusted_stats:
        row["excluded_no_cross_field_opportunity"] = len(rows) - len(adjusted_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "per_example_test_sfm.csv", rows)
    _write_csv(args.out_dir / "paired_raw_test_stats.csv", raw_stats)
    _write_csv(args.out_dir / "paired_adjusted_test_stats.csv", adjusted_stats)
    provenance = {
        "scope": "One-time confirmatory analysis on untouched DBpedia benchmark test records",
        "source_records": source_records,
        "selection": f"deterministic reservoir sample of {args.sample_size} records",
        "analysis_seed": args.analysis_seed,
        "test_jsonl": str(args.test_jsonl),
        "test_jsonl_sha256": _sha256(args.test_jsonl),
        "test_source_csv_sha256": manifest["source_csv_sha256"],
        "baseline_checkpoint_sha256": _sha256(args.baseline_checkpoint),
        "saab_checkpoint_sha256": _sha256(args.saab_checkpoint),
        "checkpoint_seed": 1001,
        "checkpoint_step": 500,
        "evaluation_max_length": 256,
        "named_field_ids": named_fields,
        "sample_indices": sorted(sample_indices),
        "bootstrap_resamples": args.bootstrap_resamples,
        "permutations": args.permutations,
        "interpretation_boundary": (
            "This evaluates new records for one fixed trained model pair; it is not "
            "another independent initialization."
        ),
    }
    (args.out_dir / "analysis.json").write_text(
        json.dumps(
            {
                "provenance": provenance,
                "raw_paired_stats": raw_stats,
                "opportunity_adjusted_paired_stats": adjusted_stats,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved untouched-test analysis to {args.out_dir}", flush=True)
    return args.out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--saab-checkpoint", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--field-vocab-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=2336)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--analysis-seed", type=int, default=1001)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
