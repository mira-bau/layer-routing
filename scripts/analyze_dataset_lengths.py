"""Compare tokenized length distributions in prepared DBpedia and PubMed data.

The script streams JSONL records directly, including DBpedia records stored in
a ZIP archive. It reports observed post-preparation lengths. Because the
prepared artifacts do not retain pre-truncation lengths, the fraction at the
configured maximum is reported as maximum-length saturation, not as an exact
truncation rate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator


DBPEDIA_PREFIX = "data/processed/benchmark/dbpedia_msm"


def _iter_jsonl(lines: Iterable[bytes | str], source: str) -> Iterator[dict[str, Any]]:
    for line_number, raw_line in enumerate(lines, start=1):
        if isinstance(raw_line, bytes):
            raw_line = raw_line.decode("utf-8")
        if not raw_line.strip():
            continue
        try:
            yield json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {source} at line {line_number}: {exc}") from exc


def _record_length(record: dict[str, Any], source: str, index: int) -> int:
    required = ("input_ids", "field_ids", "attention_mask")
    missing = [name for name in required if name not in record]
    if missing:
        raise ValueError(f"{source} record {index} is missing: {', '.join(missing)}")
    input_ids = record["input_ids"]
    field_ids = record["field_ids"]
    attention_mask = record["attention_mask"]
    if not (len(input_ids) == len(field_ids) == len(attention_mask)):
        raise ValueError(f"{source} record {index} has misaligned tensor lengths")
    if any(value not in (0, 1) for value in attention_mask):
        raise ValueError(f"{source} record {index} has a non-binary attention mask")
    length = int(sum(attention_mask))
    if length <= 0:
        raise ValueError(f"{source} record {index} has no visible tokens")
    return length


def _collect_lengths(
    records: Iterable[dict[str, Any]],
    *,
    dataset: str,
    split: str,
    source: str,
    configured_max_length: int,
    expected_records: int | None,
) -> tuple[list[int], dict[str, Any]]:
    lengths: list[int] = []
    for index, record in enumerate(records, start=1):
        length = _record_length(record, source, index)
        if length > configured_max_length:
            raise ValueError(
                f"{source} record {index} length {length} exceeds configured maximum "
                f"{configured_max_length}"
            )
        lengths.append(length)
        if index % 50_000 == 0:
            print(f"length-analysis dataset={dataset} split={split} records={index}", flush=True)
    if expected_records is not None and len(lengths) != expected_records:
        raise ValueError(
            f"{source} contains {len(lengths)} records; manifest declares {expected_records}"
        )
    return lengths, _summarize(dataset, split, source, configured_max_length, lengths)


def _linear_quantile(sorted_values: list[int], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute a quantile of an empty sequence")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _summarize(
    dataset: str,
    split: str,
    source: str,
    configured_max_length: int,
    lengths: list[int],
) -> dict[str, Any]:
    if not lengths:
        raise ValueError(f"No records found in {source}")
    ordered = sorted(lengths)
    saturated = sum(length == configured_max_length for length in lengths)
    return {
        "dataset": dataset,
        "split": split,
        "source": source,
        "records": len(lengths),
        "configured_max_length": configured_max_length,
        "minimum": min(lengths),
        "mean": statistics.fmean(lengths),
        "population_standard_deviation": statistics.pstdev(lengths),
        "p25": _linear_quantile(ordered, 0.25),
        "median": _linear_quantile(ordered, 0.50),
        "p75": _linear_quantile(ordered, 0.75),
        "p90": _linear_quantile(ordered, 0.90),
        "p95": _linear_quantile(ordered, 0.95),
        "maximum": max(lengths),
        "maximum_length_saturation_count": saturated,
        "maximum_length_saturation_rate": saturated / len(lengths),
        "exact_truncation_rate": None,
        "exact_truncation_rate_note": (
            "Unavailable because prepared records do not retain pre-truncation lengths. "
            "Maximum-length saturation is an upper bound on the exact truncation rate."
        ),
        "quantile_method": "linear interpolation at (n - 1) * probability",
    }


def _read_dbpedia(
    archive_path: Path, splits: list[str]
) -> tuple[dict[str, list[int]], list[dict[str, Any]], dict[str, Any]]:
    lengths_by_split: dict[str, list[int]] = {}
    summaries: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        manifest_name = f"{DBPEDIA_PREFIX}/manifest.json"
        manifest = json.loads(archive.read(manifest_name))
        for split in splits:
            entry = f"{DBPEDIA_PREFIX}/{split}.jsonl"
            if entry not in archive.namelist():
                raise FileNotFoundError(f"Missing ZIP entry: {entry}")
            split_manifest = manifest["splits"][split]
            with archive.open(entry) as handle:
                lengths, summary = _collect_lengths(
                    _iter_jsonl(handle, f"{archive_path}:{entry}"),
                    dataset="DBpedia",
                    split=split,
                    source=f"{archive_path}:{entry}",
                    configured_max_length=int(manifest["max_length"]),
                    expected_records=int(split_manifest["records"]),
                )
            lengths_by_split[split] = lengths
            summaries.append(summary)
    provenance = {
        "dataset": "DBpedia",
        "container": str(archive_path),
        "container_sha256": _sha256_file(archive_path),
        "manifest": manifest,
    }
    return lengths_by_split, summaries, provenance


def _read_pubmed(
    prepared_dir: Path, splits: list[str]
) -> tuple[dict[str, list[int]], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = prepared_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    lengths_by_split: dict[str, list[int]] = {}
    summaries: list[dict[str, Any]] = []
    file_hashes = {"manifest.json": _sha256_file(manifest_path)}
    for split in splits:
        path = prepared_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        split_manifest = manifest["splits"][split]
        digest = hashlib.sha256()

        def hashing_lines() -> Iterator[bytes]:
            with path.open("rb") as handle:
                for line in handle:
                    digest.update(line)
                    yield line

        lengths, summary = _collect_lengths(
            _iter_jsonl(hashing_lines(), str(path)),
            dataset="PubMed",
            split=split,
            source=str(path),
            configured_max_length=int(manifest["max_length"]),
            expected_records=int(split_manifest["records"]),
        )
        file_hashes[f"{split}.jsonl"] = digest.hexdigest()
        lengths_by_split[split] = lengths
        summaries.append(summary)
    provenance = {
        "dataset": "PubMed",
        "prepared_dir": str(prepared_dir),
        "file_sha256": file_hashes,
        "manifest": manifest,
    }
    return lengths_by_split, summaries, provenance


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_frequency_csv(
    path: Path,
    dbpedia: dict[str, list[int]],
    pubmed: dict[str, list[int]],
) -> None:
    rows = []
    for dataset, split_map in (("DBpedia", dbpedia), ("PubMed", pubmed)):
        for split, lengths in split_map.items():
            counts = Counter(lengths)
            for length in sorted(counts):
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "token_length": length,
                        "count": counts[length],
                        "proportion": counts[length] / len(lengths),
                    }
                )
    _write_csv(path, rows)


def _plot_validation_lengths(
    path_png: Path,
    path_pdf: Path,
    dbpedia_lengths: list[int],
    pubmed_lengths: list[int],
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib and numpy") from exc

    colors = {"DBpedia": "#2C7BB6", "PubMed": "#D95F02"}
    data = {"DBpedia": dbpedia_lengths, "PubMed": pubmed_lengths}
    bins = np.arange(0, 529, 16)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for dataset, values in data.items():
        axes[0].hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            color=colors[dataset],
            label=dataset,
        )
        ordered = np.sort(np.asarray(values))
        cumulative = np.arange(1, len(ordered) + 1) / len(ordered)
        axes[1].plot(ordered, cumulative, linewidth=2.0, color=colors[dataset], label=dataset)

    axes[0].set_title("Validation token-length distribution")
    axes[0].set_xlabel("Observed tokenized length")
    axes[0].set_ylabel("Density")
    axes[1].set_title("Validation token-length ECDF")
    axes[1].set_xlabel("Observed tokenized length")
    axes[1].set_ylabel("Cumulative proportion")
    for axis in axes:
        axis.set_xlim(0, 520)
        axis.grid(True, alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path_png, dpi=180, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def run(args) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = list(args.splits)
    dbpedia, dbpedia_summaries, dbpedia_provenance = _read_dbpedia(
        Path(args.dbpedia_zip), splits
    )
    pubmed, pubmed_summaries, pubmed_provenance = _read_pubmed(
        Path(args.pubmed_prepared_dir), splits
    )
    summaries = dbpedia_summaries + pubmed_summaries
    payload = {
        "analysis_scope": (
            "Observed post-preparation token lengths for exact prepared train and validation splits."
        ),
        "interpretation_boundary": (
            "This dataset-only analysis cannot establish whether sequence length explains "
            "Baseline-versus-SAAB routing differences."
        ),
        "summaries": summaries,
    }
    (out_dir / "length_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_csv(out_dir / "length_summary.csv", summaries)
    _write_frequency_csv(out_dir / "length_frequencies.csv", dbpedia, pubmed)
    (out_dir / "provenance.json").write_text(
        json.dumps(
            {"dbpedia": dbpedia_provenance, "pubmed": pubmed_provenance}, indent=2
        )
        + "\n"
    )
    _plot_validation_lengths(
        out_dir / "validation_length_distributions.png",
        out_dir / "validation_length_distributions.pdf",
        dbpedia["val"],
        pubmed["val"],
    )
    print(f"length-analysis complete output={out_dir} summary_rows={len(summaries)}")
    return out_dir


def _parse_splits(value: str) -> list[str]:
    splits = [item.strip() for item in value.split(",") if item.strip()]
    if not splits or any(split not in {"train", "val"} for split in splits):
        raise argparse.ArgumentTypeError("splits must contain train and/or val")
    return splits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbpedia-zip", type=Path, required=True)
    parser.add_argument("--pubmed-prepared-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--splits", type=_parse_splits, default=["train", "val"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
