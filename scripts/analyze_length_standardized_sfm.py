#!/usr/bin/env python3
"""Standardize DBpedia and PubMed SFM changes to shared token lengths."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["in_length_sample"] != "True":
                continue
            rows.append(
                {
                    "length": int(row["evaluation_token_length"]),
                    "delta": np.asarray(
                        [float(row[f"delta_l{layer}"]) for layer in range(4)]
                    ),
                }
            )
    if not rows:
        raise ValueError(f"No length-sample rows found in {path}")
    return rows


def _group(rows: list[dict[str, object]], lower: int, width: int):
    groups: dict[int, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        key = (int(row["length"]) - lower) // width
        groups[key].append(row["delta"])
    return {key: np.stack(values) for key, values in groups.items()}


def _standardize(
    dbpedia: list[dict[str, object]],
    pubmed: list[dict[str, object]],
    *,
    lower: int,
    upper: int,
    width: int,
) -> dict[str, object]:
    db = _group([row for row in dbpedia if lower <= row["length"] <= upper], lower, width)
    pm = _group([row for row in pubmed if lower <= row["length"] <= upper], lower, width)
    shared = sorted(db.keys() & pm.keys())
    weights = {key: min(len(db[key]), len(pm[key])) for key in shared}
    total_weight = sum(weights.values())
    db_mean = sum(weights[key] * db[key].mean(axis=0) for key in shared) / total_weight
    pm_mean = sum(weights[key] * pm[key].mean(axis=0) for key in shared) / total_weight
    db_length = sum(
        weights[key] * np.mean([row["length"] for row in dbpedia if lower <= row["length"] <= upper and (row["length"] - lower) // width == key])
        for key in shared
    ) / total_weight
    pm_length = sum(
        weights[key] * np.mean([row["length"] for row in pubmed if lower <= row["length"] <= upper and (row["length"] - lower) // width == key])
        for key in shared
    ) / total_weight
    return {
        "band_width": width,
        "shared_bands": len(shared),
        "overlap_target_mass_per_dataset": total_weight,
        "standardized_mean_length": {"dbpedia": db_length, "pubmed": pm_length},
        "dbpedia_standardized_delta": db_mean.tolist(),
        "pubmed_standardized_delta": pm_mean.tolist(),
        "pubmed_minus_dbpedia": (pm_mean - db_mean).tolist(),
        "groups": {"dbpedia": db, "pubmed": pm, "weights": weights},
    }


def run(args: argparse.Namespace) -> Path:
    dbpedia = _load(args.dbpedia_csv)
    pubmed = _load(args.pubmed_csv)
    lower = max(min(row["length"] for row in dbpedia), min(row["length"] for row in pubmed))
    upper = min(max(row["length"] for row in dbpedia), max(row["length"] for row in pubmed))
    analyses = [
        _standardize(dbpedia, pubmed, lower=lower, upper=upper, width=width)
        for width in args.band_widths
    ]

    exact = analyses[0]
    if exact["band_width"] != 1:
        raise ValueError("The first band width must be 1 for exact-length inference")
    groups = exact.pop("groups")
    rng = np.random.default_rng(args.seed)
    bootstrap = np.zeros((args.bootstrap_resamples, 4))
    weights = groups["weights"]
    total_weight = exact["overlap_target_mass_per_dataset"]
    for sample in range(args.bootstrap_resamples):
        contrast = np.zeros(4)
        for key, weight in weights.items():
            db = groups["dbpedia"][key]
            pm = groups["pubmed"][key]
            db_mean = db[rng.integers(0, len(db), len(db))].mean(axis=0)
            pm_mean = pm[rng.integers(0, len(pm), len(pm))].mean(axis=0)
            contrast += weight * (pm_mean - db_mean)
        bootstrap[sample] = contrast / total_weight
    exact["contrast_bootstrap_ci95"] = [
        {
            "layer": layer,
            "low": float(np.quantile(bootstrap[:, layer], 0.025)),
            "high": float(np.quantile(bootstrap[:, layer], 0.975)),
        }
        for layer in range(4)
    ]
    for analysis in analyses[1:]:
        analysis.pop("groups")

    payload = {
        "provenance": {
            "dbpedia_csv": str(args.dbpedia_csv),
            "dbpedia_csv_sha256": _sha256(args.dbpedia_csv),
            "pubmed_csv": str(args.pubmed_csv),
            "pubmed_csv_sha256": _sha256(args.pubmed_csv),
            "pubmed_evaluation_max_length": 512,
            "shared_length_minimum": lower,
            "shared_length_maximum": upper,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
        },
        "exact_length_standardization": exact,
        "band_width_sensitivity": analyses[1:],
        "interpretation_boundary": (
            "Standardization estimates dataset differences only within observed shared "
            "length support; it does not establish a causal effect of dataset identity "
            "or generalize to PubMed lengths absent from DBpedia."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"length-standardized analysis complete output={args.output}")
    return args.output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbpedia-csv", required=True, type=Path)
    parser.add_argument("--pubmed-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--band-widths", default="1,2,5,10")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1001)
    args = parser.parse_args()
    args.band_widths = [int(value) for value in args.band_widths.split(",")]
    if not args.band_widths or any(width <= 0 for width in args.band_widths):
        raise ValueError("Band widths must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
