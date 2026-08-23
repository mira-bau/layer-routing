#!/usr/bin/env python3
"""Build the initialization-analysis outcome CSV from attention diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("seeds must not be empty")
    if len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must not contain duplicates")
    return seeds


def _variant(payload: dict[str, Any], name: str, seed: int, expected_step: int) -> list[float]:
    variant = payload.get(name)
    if not isinstance(variant, dict):
        raise ValueError(f"diagnostic is missing the {name} result")
    if int(variant.get("seed", -1)) != seed:
        raise ValueError(
            f"{name} diagnostic seed is {variant.get('seed')!r}; expected {seed}"
        )
    if int(variant.get("step", -1)) != expected_step:
        raise ValueError(
            f"{name} diagnostic step is {variant.get('step')!r}; "
            f"expected {expected_step}"
        )
    values = variant.get("same_field_mass_per_layer")
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError(f"{name} diagnostic must contain four layer-wise SFM values")
    return [float(value) for value in values]


def build_rows(
    diagnostics_root: Path,
    seeds: list[int],
    *,
    expected_step: int,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for seed in seeds:
        path = diagnostics_root / f"seed{seed}" / "attention_diagnostics.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        baseline = _variant(payload, "baseline", seed, expected_step)
        saab = _variant(payload, "saab", seed, expected_step)
        rows.append(
            {
                "seed": seed,
                "baseline_l2": baseline[2],
                "baseline_l3": baseline[3],
                "saab_l2": saab[2],
                "saab_l3": saab[3],
                "delta_l2": saab[2] - baseline[2],
                "delta_l3": saab[3] - baseline[3],
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=_parse_seeds("0,7,42,99,123,256,1001,2024"),
    )
    parser.add_argument("--expected-step", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = build_rows(
        args.diagnostics_root,
        args.seeds,
        expected_step=args.expected_step,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} seed outcomes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
