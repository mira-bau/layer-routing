#!/usr/bin/env python3
"""Opportunity-normalize retained per-example same-field mass results.

This analysis does not run a model. It joins retained per-example SFM values
to the exact prepared field-ID sequences and computes the same-field mass that
uniform attention would produce given the available same-field and cross-field
keys. Chance-adjusted SFM is (SFM - opportunity) / (1 - opportunity).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import zipfile
from pathlib import Path
from typing import Any, Iterable


DBPEDIA_MEMBER = "data/processed/benchmark/dbpedia_msm/val.jsonl"


def opportunity_reference(
    field_ids: list[int],
    attention_mask: list[int],
    named_field_ids: set[int],
    evaluation_max_length: int,
) -> float:
    """Return uniform-attention same-field opportunity for one example."""
    counts = {field_id: 0 for field_id in named_field_ids}
    for field_id, is_valid in zip(
        field_ids[:evaluation_max_length], attention_mask[:evaluation_max_length]
    ):
        if is_valid and field_id in counts:
            counts[field_id] += 1
    total = sum(counts.values())
    if total == 0:
        raise ValueError("example has no valid named-field positions")
    return sum(count * count for count in counts.values()) / (total * total)


def adjusted_sfm(sfm: float, opportunity: float) -> float | None:
    """Map uniform-attention opportunity to 0 and all-same-field mass to 1."""
    denominator = 1.0 - opportunity
    if denominator <= 0.0:
        return None
    return (sfm - opportunity) / denominator


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _records_from_lines(
    lines: Iterable[bytes | str], wanted_indices: set[int]
) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for index, line in enumerate(lines):
        if index not in wanted_indices:
            continue
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        records[index] = json.loads(line)
        if len(records) == len(wanted_indices):
            break
    missing = wanted_indices.difference(records)
    if missing:
        preview = sorted(missing)[:10]
        raise ValueError(f"prepared validation data are missing indices: {preview}")
    return records


def _load_dbpedia_records(path: Path, wanted_indices: set[int]):
    with zipfile.ZipFile(path) as archive, archive.open(DBPEDIA_MEMBER) as handle:
        return _records_from_lines(handle, wanted_indices)


def _load_jsonl_records(path: Path, wanted_indices: set[int]):
    with path.open(encoding="utf-8") as handle:
        return _records_from_lines(handle, wanted_indices)


def _bootstrap_ci(values: list[float], *, resamples: int, seed: int):
    rng = random.Random(seed)
    size = len(values)
    means = sorted(
        sum(values[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(resamples)
    )
    low = means[math.floor(0.025 * (resamples - 1))]
    high = means[math.ceil(0.975 * (resamples - 1))]
    return low, high


def _sign_flip_p(values: list[float], *, permutations: int, seed: int):
    rng = random.Random(seed)
    observed = abs(sum(values) / len(values))
    extreme = 0
    for _ in range(permutations):
        permuted = abs(
            sum(value if rng.getrandbits(1) else -value for value in values)
            / len(values)
        )
        extreme += permuted >= observed
    return (extreme + 1) / (permutations + 1)


def _holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def _analyze_file(
    *,
    dataset: str,
    source_csv: Path,
    prepared_records: dict[int, dict[str, Any]],
    named_field_ids: set[int],
    output_csv: Path,
    bootstrap_resamples: int,
    permutations: int,
    analysis_seed: int,
) -> list[dict[str, Any]]:
    rows = _load_rows(source_csv)
    fieldnames = list(rows[0]) + ["opportunity_reference"]
    for layer in range(4):
        fieldnames.extend(
            [
                f"baseline_adjusted_l{layer}",
                f"saab_adjusted_l{layer}",
                f"delta_adjusted_l{layer}",
            ]
        )

    enriched: list[dict[str, Any]] = []
    for row in rows:
        source_index = int(row["source_index"])
        record = prepared_records[source_index]
        evaluation_max_length = int(row["evaluation_max_length"])
        opportunity = opportunity_reference(
            record["field_ids"],
            record["attention_mask"],
            named_field_ids,
            evaluation_max_length,
        )
        if sum(
            1
            for field_id, valid in zip(
                record["field_ids"][:evaluation_max_length],
                record["attention_mask"][:evaluation_max_length],
            )
            if valid and field_id in named_field_ids
        ) != int(row["evaluation_token_length"]):
            raise ValueError(
                f"evaluation length mismatch for {dataset} source index {source_index}"
            )
        result: dict[str, Any] = dict(row)
        result["opportunity_reference"] = opportunity
        for layer in range(4):
            baseline = adjusted_sfm(float(row[f"baseline_l{layer}"]), opportunity)
            saab = adjusted_sfm(float(row[f"saab_l{layer}"]), opportunity)
            result[f"baseline_adjusted_l{layer}"] = "" if baseline is None else baseline
            result[f"saab_adjusted_l{layer}"] = "" if saab is None else saab
            result[f"delta_adjusted_l{layer}"] = (
                "" if baseline is None or saab is None else saab - baseline
            )
        enriched.append(result)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched)

    primary_all = [row for row in enriched if row["in_primary_first64"] == "True"]
    if len(primary_all) != 64:
        raise ValueError(
            f"expected 64 primary {dataset} examples, found {len(primary_all)}"
        )
    primary = [row for row in primary_all if row["delta_adjusted_l0"] != ""]
    excluded = len(primary_all) - len(primary)
    if not primary:
        raise ValueError(f"no primary {dataset} examples have cross-field opportunity")
    raw_p_values = []
    summary = []
    for layer in range(4):
        differences = [float(row[f"delta_adjusted_l{layer}"]) for row in primary]
        low, high = _bootstrap_ci(
            differences,
            resamples=bootstrap_resamples,
            seed=analysis_seed + layer,
        )
        p_value = _sign_flip_p(
            differences,
            permutations=permutations,
            seed=analysis_seed + 100 + layer,
        )
        raw_p_values.append(p_value)
        summary.append(
            {
                "dataset": dataset,
                "evaluation_max_length": int(primary[0]["evaluation_max_length"]),
                "examples": len(primary),
                "excluded_no_cross_field_opportunity": excluded,
                "layer": layer,
                "mean_opportunity_reference": sum(
                    float(row["opportunity_reference"]) for row in primary_all
                )
                / len(primary_all),
                "mean_baseline_adjusted_sfm": sum(
                    float(row[f"baseline_adjusted_l{layer}"]) for row in primary
                )
                / len(primary),
                "mean_saab_adjusted_sfm": sum(
                    float(row[f"saab_adjusted_l{layer}"]) for row in primary
                )
                / len(primary),
                "mean_paired_adjusted_difference": sum(differences) / len(differences),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "permutation_p_two_sided_raw": p_value,
                "bootstrap_resamples": bootstrap_resamples,
                "permutations": permutations,
            }
        )
    for row, adjusted_p in zip(summary, _holm(raw_p_values)):
        row["permutation_p_holm_four_layers"] = adjusted_p
    return summary


def _write_summary(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--dbpedia-zip", type=Path, required=True)
    parser.add_argument("--pubmed-validation-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--analysis-seed", type=int, default=1001)
    args = parser.parse_args()

    specifications = [
        (
            "DBpedia",
            args.analysis_dir / "dbpedia" / "per_example_sfm.csv",
            {3, 4},
            _load_dbpedia_records,
            args.dbpedia_zip,
            "dbpedia_adjusted_per_example.csv",
        ),
        (
            "PubMed-256",
            args.analysis_dir / "pubmed" / "per_example_sfm.csv",
            {3, 4, 5},
            _load_jsonl_records,
            args.pubmed_validation_jsonl,
            "pubmed_256_adjusted_per_example.csv",
        ),
        (
            "PubMed-512",
            args.analysis_dir / "pubmed" / "configured_length_per_example_sfm.csv",
            {3, 4, 5},
            _load_jsonl_records,
            args.pubmed_validation_jsonl,
            "pubmed_512_adjusted_per_example.csv",
        ),
    ]

    all_summary: list[dict[str, Any]] = []
    for dataset, source_csv, field_ids, loader, prepared_path, output_name in specifications:
        rows = _load_rows(source_csv)
        wanted_indices = {int(row["source_index"]) for row in rows}
        records = loader(prepared_path, wanted_indices)
        summary = _analyze_file(
            dataset=dataset,
            source_csv=source_csv,
            prepared_records=records,
            named_field_ids=field_ids,
            output_csv=args.out_dir / output_name,
            bootstrap_resamples=args.bootstrap_resamples,
            permutations=args.permutations,
            analysis_seed=args.analysis_seed,
        )
        all_summary.extend(summary)
        print(f"opportunity-sfm dataset={dataset} examples={len(rows)} complete", flush=True)

    _write_summary(args.out_dir / "opportunity_adjusted_primary_stats.csv", all_summary)
    metadata = {
        "definition": {
            "opportunity_reference": "sum_f(n_f^2) / N^2 over valid named-field positions",
            "adjusted_sfm": "(SFM - opportunity_reference) / (1 - opportunity_reference)",
            "interpretation": "0 is uniform attention over valid keys; 1 is exclusively same-field attention; negative values indicate more cross-field mass than the uniform-opportunity reference.",
        },
        "bootstrap_resamples": args.bootstrap_resamples,
        "permutations": args.permutations,
        "analysis_seed": args.analysis_seed,
        "model_inference_required": False,
    }
    (args.out_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved opportunity-normalized analysis to {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
