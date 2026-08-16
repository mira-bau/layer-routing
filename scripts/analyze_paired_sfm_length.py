"""Run paired per-example SFM statistics and sequence-length analyses.

This reviewer analysis evaluates exact seed-1001 Baseline/SAAB checkpoint pairs
for DBpedia and PubMed. The first 64 validation records reproduce the submitted
aggregate comparison for paired inference. A larger deterministic validation
sample is used separately for length correlations and stratified summaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    baseline_checkpoint: Path
    saab_checkpoint: Path
    validation_jsonl: Path
    field_vocab_json: Path
    expected_step: int
    expected_max_length: int
    expected_num_labels: int
    expected_field_vocab_size: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _verify_checkpoint(checkpoint: dict[str, Any], spec: DatasetSpec, variant: str) -> None:
    config = checkpoint.get("config", {})
    data = config.get("data", {})
    model = config.get("model_config", {})
    training = config.get("training", {})
    expected = {
        "variant": variant,
        "seed": 1001,
        "step": spec.expected_step,
        "max_steps": spec.expected_step,
        "max_length": spec.expected_max_length,
        "num_labels": spec.expected_num_labels,
        "field_vocab_size": spec.expected_field_vocab_size,
        "d_model": 768,
        "num_layers": 4,
        "num_heads": 6,
        "ff_dim": 3072,
        "dropout": 0.2,
        "scale_embeddings": True,
        "microbatch_size": 64,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1.0e-4,
        "lr_schedule": "linear_warmup_cosine",
        "warmup_steps": 50,
    }
    observed = {
        "variant": config.get("model"),
        "seed": config.get("seed"),
        "step": checkpoint.get("step"),
        "max_steps": training.get("max_steps"),
        "max_length": data.get("max_length"),
        "num_labels": data.get("num_labels"),
        "field_vocab_size": data.get("field_vocab_size"),
        "d_model": model.get("d_model"),
        "num_layers": model.get("num_layers"),
        "num_heads": model.get("num_heads"),
        "ff_dim": model.get("ff_dim"),
        "dropout": model.get("dropout"),
        "scale_embeddings": model.get("scale_embeddings", False),
        "microbatch_size": training.get("microbatch_size"),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps"),
        "learning_rate": training.get("learning_rate"),
        "lr_schedule": training.get("lr_schedule"),
        "warmup_steps": training.get("warmup_steps"),
    }
    mismatches = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if observed[key] != expected[key]
    }
    if mismatches:
        raise ValueError(f"{spec.name} {variant} checkpoint protocol mismatch: {mismatches}")


def _rebuild_model(checkpoint: dict[str, Any], device):
    from scripts.diag_attention import _rebuild_model

    return _rebuild_model(checkpoint).to(device).eval()


def _field_ids(field_vocab_path: Path, mask_field_id: int) -> list[int]:
    payload = json.loads(field_vocab_path.read_text())
    ids = sorted(
        int(value)
        for value in payload["field_vocab"].values()
        if int(value) > 2 and int(value) != mask_field_id
    )
    if not ids:
        raise ValueError(f"No named field IDs found in {field_vocab_path}")
    return ids


def _select_records(
    path: Path,
    *,
    primary_size: int,
    length_sample_size: int,
    sample_seed: int,
) -> tuple[list[dict[str, Any]], set[int], set[int], int]:
    primary: list[tuple[int, dict[str, Any]]] = []
    reservoir: list[tuple[int, dict[str, Any]]] = []
    rng = random.Random(sample_seed)
    source_records = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            index = source_records
            source_records += 1
            _validate_record(record, path, index)
            if len(primary) < primary_size:
                primary.append((index, record))
            if len(reservoir) < length_sample_size:
                reservoir.append((index, record))
            else:
                replacement = rng.randrange(source_records)
                if replacement < length_sample_size:
                    reservoir[replacement] = (index, record)
    if source_records < max(primary_size, length_sample_size):
        raise ValueError(
            f"{path} has {source_records} records, fewer than the requested sample size"
        )
    primary_indices = {index for index, _ in primary}
    length_indices = {index for index, _ in reservoir}
    union = {index: record for index, record in primary + reservoir}
    selected = [
        {"source_index": index, **record}
        for index, record in sorted(union.items())
    ]
    return selected, primary_indices, length_indices, source_records


def _validate_record(record: dict[str, Any], path: Path, index: int) -> None:
    arrays = [record.get(name) for name in ("input_ids", "field_ids", "attention_mask")]
    if any(not isinstance(value, list) for value in arrays):
        raise ValueError(f"{path} record {index} is missing prepared tensor lists")
    if len({len(value) for value in arrays}) != 1:
        raise ValueError(f"{path} record {index} has misaligned tensor lists")


def _batch_records(records: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def _collate(records: list[dict[str, Any]], device, evaluation_max_length: int):
    import torch

    max_length = min(
        max(len(record["input_ids"]) for record in records), evaluation_max_length
    )
    input_ids = torch.zeros(len(records), max_length, dtype=torch.long, device=device)
    field_ids = torch.zeros(len(records), max_length, dtype=torch.long, device=device)
    attention_mask = torch.zeros(len(records), max_length, dtype=torch.bool, device=device)
    for row, record in enumerate(records):
        length = min(len(record["input_ids"]), evaluation_max_length)
        input_ids[row, :length] = torch.tensor(record["input_ids"][:length], device=device)
        field_ids[row, :length] = torch.tensor(record["field_ids"][:length], device=device)
        attention_mask[row, :length] = torch.tensor(
            record["attention_mask"][:length], dtype=torch.bool, device=device
        )
    return input_ids, field_ids, attention_mask


def _per_example_sfm(attentions, field_ids, attention_mask, named_field_ids: list[int]):
    import torch

    valid = torch.zeros_like(field_ids, dtype=torch.bool)
    for field_id in named_field_ids:
        valid |= field_ids.eq(field_id)
    valid &= attention_mask.bool()
    same = field_ids.unsqueeze(2).eq(field_ids.unsqueeze(1)).unsqueeze(1)
    key_valid = valid.unsqueeze(1).unsqueeze(2)
    query_valid = valid.to(dtype=torch.float32).unsqueeze(1)
    query_count = query_valid.sum(dim=-1).clamp_min(1.0)
    values = []
    for attention in attentions:
        same_mass = (attention * key_valid * same).sum(dim=-1)
        per_head = (same_mass * query_valid).sum(dim=-1) / query_count
        values.append(per_head.mean(dim=1))
    return torch.stack(values, dim=1)


def _evaluate_model(
    model,
    records: list[dict[str, Any]],
    *,
    named_field_ids: list[int],
    batch_size: int,
    device,
    dataset: str,
    variant: str,
    evaluation_max_length: int,
) -> dict[int, list[float]]:
    import torch

    results: dict[int, list[float]] = {}
    with torch.inference_mode():
        for batch_number, batch in enumerate(_batch_records(records, batch_size), start=1):
            input_ids, field_ids, attention_mask = _collate(
                batch, device, evaluation_max_length
            )
            output = model(
                input_ids, field_ids, attention_mask=attention_mask, need_weights=True
            )
            sfm = _per_example_sfm(
                output.attentions, field_ids, attention_mask, named_field_ids
            ).detach().cpu().tolist()
            for record, values in zip(batch, sfm):
                results[int(record["source_index"])] = [float(value) for value in values]
            if batch_number == 1 or batch_number % 100 == 0:
                print(
                    f"paired-sfm dataset={dataset} variant={variant} "
                    f"evaluation_max_length={evaluation_max_length} "
                    f"batch={batch_number} evaluated={len(results)}/{len(records)} device={device}",
                    flush=True,
                )
    return results


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        value = min((total - rank) * p_values[index], 1.0)
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def _paired_stats(rows: list[dict[str, Any]], args, dataset: str) -> list[dict[str, Any]]:
    import numpy as np

    rng = np.random.default_rng(args.analysis_seed)
    results = []
    raw_p = []
    for layer in range(4):
        baseline = np.asarray([row[f"baseline_l{layer}"] for row in rows], dtype=float)
        saab = np.asarray([row[f"saab_l{layer}"] for row in rows], dtype=float)
        differences = saab - baseline
        boot_indices = rng.integers(0, len(rows), size=(args.bootstrap_resamples, len(rows)))
        boot_means = differences[boot_indices].mean(axis=1)
        observed = abs(float(differences.mean()))
        signs = rng.choice(
            np.asarray([-1.0, 1.0]), size=(args.permutations, len(rows)), replace=True
        )
        permuted = abs((signs * differences).mean(axis=1))
        p_value = (int((permuted >= observed).sum()) + 1) / (args.permutations + 1)
        sd = float(differences.std(ddof=1))
        result = {
            "dataset": dataset,
            "layer": layer,
            "examples": len(rows),
            "baseline_mean_sfm": float(baseline.mean()),
            "saab_mean_sfm": float(saab.mean()),
            "mean_paired_difference": float(differences.mean()),
            "median_paired_difference": float(np.median(differences)),
            "paired_difference_sd": sd,
            "paired_cohen_dz": float(differences.mean() / sd) if sd > 0 else None,
            "bootstrap_mean_difference_ci95_low": float(np.quantile(boot_means, 0.025)),
            "bootstrap_mean_difference_ci95_high": float(np.quantile(boot_means, 0.975)),
            "permutation_p_two_sided_raw": p_value,
            "bootstrap_resamples": args.bootstrap_resamples,
            "permutations": args.permutations,
        }
        results.append(result)
        raw_p.append(p_value)
    adjusted = _holm_adjust(raw_p)
    for result, value in zip(results, adjusted):
        result["permutation_p_holm_four_layers"] = value
    return results


def _length_stats(rows: list[dict[str, Any]], args, dataset: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import numpy as np
    from scipy.stats import spearmanr

    rng = np.random.default_rng(args.analysis_seed + 17)
    lengths = np.asarray([row["evaluation_token_length"] for row in rows], dtype=float)
    correlations = []
    raw_p = []
    for layer in range(4):
        differences = np.asarray([row[f"delta_l{layer}"] for row in rows], dtype=float)
        rho, p_value = spearmanr(lengths, differences)
        boot_rho = []
        for _ in range(args.correlation_bootstrap_resamples):
            indices = rng.integers(0, len(rows), size=len(rows))
            sampled_rho, _ = spearmanr(lengths[indices], differences[indices])
            if math.isfinite(float(sampled_rho)):
                boot_rho.append(float(sampled_rho))
        correlations.append(
            {
                "dataset": dataset,
                "layer": layer,
                "examples": len(rows),
                "spearman_rho": float(rho),
                "spearman_p_two_sided_raw": float(p_value),
                "spearman_rho_bootstrap_ci95_low": float(np.quantile(boot_rho, 0.025)),
                "spearman_rho_bootstrap_ci95_high": float(np.quantile(boot_rho, 0.975)),
                "bootstrap_resamples": args.correlation_bootstrap_resamples,
            }
        )
        raw_p.append(float(p_value))
    for result, value in zip(correlations, _holm_adjust(raw_p)):
        result["spearman_p_holm_four_layers"] = value

    cutpoints = [float(np.quantile(lengths, q)) for q in (0.25, 0.50, 0.75)]
    bin_rows = []
    ordered_rows = sorted(
        rows,
        key=lambda row: (row["evaluation_token_length"], row["source_index"]),
    )
    for rank, row in enumerate(ordered_rows):
        row["length_quartile_bin"] = min(4, (rank * 4) // len(ordered_rows) + 1)
    for bin_index in range(1, 5):
        subset = [row for row in rows if row["length_quartile_bin"] == bin_index]
        for layer in range(4):
            values = [row[f"delta_l{layer}"] for row in subset]
            bin_rows.append(
                {
                    "dataset": dataset,
                    "length_quartile_bin": bin_index,
                    "bin_cutpoints_p25_p50_p75": json.dumps(cutpoints),
                    "examples": len(subset),
                    "minimum_length": min(row["evaluation_token_length"] for row in subset),
                    "maximum_length": max(row["evaluation_token_length"] for row in subset),
                    "layer": layer,
                    "mean_paired_difference": statistics_fmean(values),
                    "median_paired_difference": float(np.median(values)),
                }
            )
    return correlations, bin_rows


def statistics_fmean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _make_rows(
    spec: DatasetSpec,
    records: list[dict[str, Any]],
    primary_indices: set[int],
    length_indices: set[int],
    baseline: dict[int, list[float]],
    saab: dict[int, list[float]],
    evaluation_max_length: int,
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        index = int(record["source_index"])
        row = {
            "dataset": spec.name,
            "source_index": index,
            "row_id": record.get("row_id"),
            "prepared_token_length": int(sum(record["attention_mask"])),
            "evaluation_token_length": min(
                int(sum(record["attention_mask"])), evaluation_max_length
            ),
            "evaluation_max_length": evaluation_max_length,
            "in_primary_first64": index in primary_indices,
            "in_length_sample": index in length_indices,
        }
        for layer in range(4):
            row[f"baseline_l{layer}"] = baseline[index][layer]
            row[f"saab_l{layer}"] = saab[index][layer]
            row[f"delta_l{layer}"] = saab[index][layer] - baseline[index][layer]
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_length_results(path_png: Path, path_pdf: Path, all_rows: dict[str, list[dict[str, Any]]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey="row")
    colors = {"DBpedia": "#2C7BB6", "PubMed": "#D95F02"}
    for row_index, dataset in enumerate(("DBpedia", "PubMed")):
        rows = [row for row in all_rows[dataset] if row["in_length_sample"]]
        lengths = np.asarray([row["evaluation_token_length"] for row in rows])
        for layer in range(4):
            deltas = np.asarray([row[f"delta_l{layer}"] for row in rows])
            axis = axes[row_index, layer]
            axis.scatter(lengths, deltas, s=8, alpha=0.18, color=colors[dataset], rasterized=True)
            order = np.argsort(lengths)
            window = max(25, len(rows) // 20)
            kernel = np.ones(window) / window
            smooth_x = np.convolve(lengths[order], kernel, mode="valid")
            smooth_y = np.convolve(deltas[order], kernel, mode="valid")
            axis.plot(smooth_x, smooth_y, color="black", linewidth=1.5)
            axis.axhline(0.0, color="gray", linewidth=0.8)
            axis.set_title(f"{dataset}, L{layer}")
            axis.set_xlabel("Tokenized length")
            if layer == 0:
                axis.set_ylabel("SAAB − Baseline SFM")
            axis.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path_png, dpi=180, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def _common_length_range_summary(
    all_rows: dict[str, list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    sampled = {
        dataset: [row for row in rows if row["in_length_sample"]]
        for dataset, rows in all_rows.items()
    }
    lower = max(
        min(row["evaluation_token_length"] for row in rows) for rows in sampled.values()
    )
    upper = min(
        max(row["evaluation_token_length"] for row in rows) for rows in sampled.values()
    )
    summaries = []
    for dataset, rows in sampled.items():
        subset = [
            row for row in rows
            if lower <= row["evaluation_token_length"] <= upper
        ]
        for layer in range(4):
            values = [row[f"delta_l{layer}"] for row in subset]
            summaries.append(
                {
                    "dataset": dataset,
                    "common_length_minimum": lower,
                    "common_length_maximum": upper,
                    "examples": len(subset),
                    "layer": layer,
                    "mean_paired_difference": statistics_fmean(values),
                    "minimum_observed_length": min(
                        row["evaluation_token_length"] for row in subset
                    ),
                    "maximum_observed_length": max(
                        row["evaluation_token_length"] for row in subset
                    ),
                }
            )
    return summaries, {"minimum": lower, "maximum": upper}


def _evaluate_dataset(spec: DatasetSpec, args, device, out_dir: Path):
    import torch

    baseline_checkpoint = _load_checkpoint(spec.baseline_checkpoint)
    saab_checkpoint = _load_checkpoint(spec.saab_checkpoint)
    _verify_checkpoint(baseline_checkpoint, spec, "baseline")
    _verify_checkpoint(saab_checkpoint, spec, "saab")
    records, primary_indices, length_indices, source_records = _select_records(
        spec.validation_jsonl,
        primary_size=args.primary_examples,
        length_sample_size=args.length_sample_size,
        sample_seed=args.analysis_seed,
    )
    mask_field_id = int(baseline_checkpoint["config"]["data"]["mask_field_id"])
    named_fields = _field_ids(spec.field_vocab_json, mask_field_id)
    submitted_diagnostic_max_length = 256
    run_configured_sensitivity = spec.expected_max_length != submitted_diagnostic_max_length

    baseline_model = _rebuild_model(baseline_checkpoint, device)
    baseline = _evaluate_model(
        baseline_model,
        records,
        named_field_ids=named_fields,
        batch_size=args.eval_batch_size,
        device=device,
        dataset=spec.name,
        variant="baseline",
        evaluation_max_length=submitted_diagnostic_max_length,
    )
    baseline_configured = (
        _evaluate_model(
            baseline_model,
            records,
            named_field_ids=named_fields,
            batch_size=args.eval_batch_size,
            device=device,
            dataset=spec.name,
            variant="baseline_configured_length",
            evaluation_max_length=spec.expected_max_length,
        )
        if run_configured_sensitivity
        else None
    )
    del baseline_model, baseline_checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()

    saab_model = _rebuild_model(saab_checkpoint, device)
    saab = _evaluate_model(
        saab_model,
        records,
        named_field_ids=named_fields,
        batch_size=args.eval_batch_size,
        device=device,
        dataset=spec.name,
        variant="saab",
        evaluation_max_length=submitted_diagnostic_max_length,
    )
    saab_configured = (
        _evaluate_model(
            saab_model,
            records,
            named_field_ids=named_fields,
            batch_size=args.eval_batch_size,
            device=device,
            dataset=spec.name,
            variant="saab_configured_length",
            evaluation_max_length=spec.expected_max_length,
        )
        if run_configured_sensitivity
        else None
    )
    del saab_model, saab_checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rows = _make_rows(
        spec,
        records,
        primary_indices,
        length_indices,
        baseline,
        saab,
        submitted_diagnostic_max_length,
    )
    primary_rows = [row for row in rows if row["in_primary_first64"]]
    length_rows = [row for row in rows if row["in_length_sample"]]
    paired = _paired_stats(primary_rows, args, spec.name)
    correlations, bins = _length_stats(length_rows, args, spec.name)
    configured_sensitivity = None
    configured_rows = None
    if run_configured_sensitivity:
        configured_rows = _make_rows(
            spec,
            records,
            primary_indices,
            length_indices,
            baseline_configured,
            saab_configured,
            spec.expected_max_length,
        )
        configured_primary = [
            row for row in configured_rows if row["in_primary_first64"]
        ]
        configured_length = [
            row for row in configured_rows if row["in_length_sample"]
        ]
        configured_paired = _paired_stats(
            configured_primary, args, f"{spec.name}_configured_length"
        )
        configured_correlations, configured_bins = _length_stats(
            configured_length, args, f"{spec.name}_configured_length"
        )
        configured_sensitivity = {
            "evaluation_max_length": spec.expected_max_length,
            "paired_primary_stats": configured_paired,
            "length_correlations": configured_correlations,
            "length_quartile_bins": configured_bins,
        }
    provenance = {
        "dataset": spec.name,
        "validation_jsonl": str(spec.validation_jsonl),
        "validation_jsonl_sha256": _sha256(spec.validation_jsonl),
        "validation_source_records": source_records,
        "primary_selection": f"first {args.primary_examples} validation records",
        "length_selection": (
            f"deterministic reservoir sample of {args.length_sample_size} validation records"
        ),
        "length_sample_seed": args.analysis_seed,
        "baseline_checkpoint": str(spec.baseline_checkpoint),
        "baseline_checkpoint_sha256": _sha256(spec.baseline_checkpoint),
        "saab_checkpoint": str(spec.saab_checkpoint),
        "saab_checkpoint_sha256": _sha256(spec.saab_checkpoint),
        "checkpoint_step": spec.expected_step,
        "named_field_ids": named_fields,
        "submitted_diagnostic_max_length": submitted_diagnostic_max_length,
        "configured_model_max_length": spec.expected_max_length,
        "configured_length_sensitivity_run": run_configured_sensitivity,
    }
    dataset_dir = out_dir / spec.name.lower()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(dataset_dir / "per_example_sfm.csv", rows)
    _write_csv(dataset_dir / "paired_primary_stats.csv", paired)
    _write_csv(dataset_dir / "length_correlations.csv", correlations)
    _write_csv(dataset_dir / "length_quartile_bins.csv", bins)
    if configured_rows is not None:
        _write_csv(dataset_dir / "configured_length_per_example_sfm.csv", configured_rows)
        _write_csv(
            dataset_dir / "configured_length_paired_primary_stats.csv",
            configured_sensitivity["paired_primary_stats"],
        )
        _write_csv(
            dataset_dir / "configured_length_correlations.csv",
            configured_sensitivity["length_correlations"],
        )
        _write_csv(
            dataset_dir / "configured_length_quartile_bins.csv",
            configured_sensitivity["length_quartile_bins"],
        )
    (dataset_dir / "analysis.json").write_text(
        json.dumps(
            {
                "provenance": provenance,
                "paired_primary_stats": paired,
                "length_correlations": correlations,
                "length_quartile_bins": bins,
                "configured_length_sensitivity": configured_sensitivity,
            },
            indent=2,
        )
        + "\n"
    )
    return rows, paired, correlations, bins, provenance, configured_sensitivity


def run(args) -> Path:
    import numpy
    import scipy
    import torch

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required. Select a Colab GPU runtime.")
        device = torch.device("cuda")
    elif args.device == "cpu" and args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("CPU use requires --device cpu --allow-cpu for testing only")

    specs = [
        DatasetSpec(
            "DBpedia",
            Path(args.dbpedia_baseline_checkpoint),
            Path(args.dbpedia_saab_checkpoint),
            Path(args.dbpedia_validation_jsonl),
            Path(args.dbpedia_field_vocab_json),
            500,
            256,
            5,
            6,
        ),
        DatasetSpec(
            "PubMed",
            Path(args.pubmed_baseline_checkpoint),
            Path(args.pubmed_saab_checkpoint),
            Path(args.pubmed_validation_jsonl),
            Path(args.pubmed_field_vocab_json),
            1500,
            512,
            6,
            7,
        ),
    ]
    for spec in specs:
        for path in (
            spec.baseline_checkpoint,
            spec.saab_checkpoint,
            spec.validation_jsonl,
            spec.field_vocab_json,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = {}
    combined_paired = []
    combined_correlations = []
    combined_bins = []
    provenance = {}
    configured_length_sensitivity = {}
    for spec in specs:
        (
            rows,
            paired,
            correlations,
            bins,
            dataset_provenance,
            dataset_sensitivity,
        ) = _evaluate_dataset(spec, args, device, out_dir)
        all_rows[spec.name] = rows
        combined_paired.extend(paired)
        combined_correlations.extend(correlations)
        combined_bins.extend(bins)
        provenance[spec.name.lower()] = dataset_provenance
        if dataset_sensitivity is not None:
            configured_length_sensitivity[spec.name.lower()] = dataset_sensitivity

    _write_csv(out_dir / "paired_primary_stats.csv", combined_paired)
    _write_csv(out_dir / "length_correlations.csv", combined_correlations)
    _write_csv(out_dir / "length_quartile_bins.csv", combined_bins)
    common_length_rows, common_length_range = _common_length_range_summary(all_rows)
    _write_csv(out_dir / "common_length_range.csv", common_length_rows)
    _plot_length_results(
        out_dir / "length_vs_paired_sfm.png",
        out_dir / "length_vs_paired_sfm.pdf",
        all_rows,
    )
    summary = {
        "analysis_scope": {
            "paired_primary": (
                f"First {args.primary_examples} held-out validation examples for each "
                "fixed seed-1001 Baseline/SAAB model pair."
            ),
            "length_analysis": (
                f"Deterministic {args.length_sample_size}-example validation sample per dataset."
            ),
            "submitted_diagnostic_length": (
                "Both submitted aggregate diagnostics used a 256-token evaluation cap."
            ),
            "pubmed_sensitivity": (
                "The same PubMed examples are additionally evaluated at the configured "
                "512-token model maximum."
            ),
        },
        "interpretation_boundary": (
            "Paired primary tests quantify validation-example variability for fixed model pairs, "
            "not variability across independently trained seeds. Length associations can show "
            "whether length covaries with routing change but cannot prove causal absence."
        ),
        "software": {
            "python_torch": torch.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "settings": {
            "primary_examples": args.primary_examples,
            "length_sample_size": args.length_sample_size,
            "eval_batch_size": args.eval_batch_size,
            "bootstrap_resamples": args.bootstrap_resamples,
            "correlation_bootstrap_resamples": args.correlation_bootstrap_resamples,
            "permutations": args.permutations,
            "analysis_seed": args.analysis_seed,
            "multiple_comparison_correction": "Holm within the four layers of each dataset",
            "alternative": "two-sided",
        },
        "provenance": provenance,
        "paired_primary_stats": combined_paired,
        "length_correlations": combined_correlations,
        "length_quartile_bins": combined_bins,
        "common_length_range": common_length_range,
        "common_length_range_summaries": common_length_rows,
        "configured_length_sensitivity": configured_length_sensitivity,
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"paired-sfm analysis complete output={out_dir}", flush=True)
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbpedia-baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--dbpedia-saab-checkpoint", type=Path, required=True)
    parser.add_argument("--dbpedia-validation-jsonl", type=Path, required=True)
    parser.add_argument("--dbpedia-field-vocab-json", type=Path, required=True)
    parser.add_argument("--pubmed-baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--pubmed-saab-checkpoint", type=Path, required=True)
    parser.add_argument("--pubmed-validation-jsonl", type=Path, required=True)
    parser.add_argument("--pubmed-field-vocab-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--primary-examples", type=int, default=64)
    parser.add_argument("--length-sample-size", type=int, default=2336)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--correlation-bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--analysis-seed", type=int, default=1001)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    positive = (
        args.primary_examples,
        args.length_sample_size,
        args.eval_batch_size,
        args.bootstrap_resamples,
        args.correlation_bootstrap_resamples,
        args.permutations,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("Sample sizes, batch size, and resampling counts must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
