#!/usr/bin/env python3
"""Reproduce the manuscript's exploratory initialization-sensitivity analysis.

The script reconstructs paired Baseline/SAAB initial states from the reported
seeds, summarizes simple parameter-scale properties, runs the one-update MSM
probe, and measures the dropout-disabled initial-loss response to SAAB over
fixed validation blocks and deterministic MSM masks. It reads local prepared
data and a caller-supplied CSV of final routing outcomes; no reported result is
embedded in the implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SEEDS = (0, 7, 42, 99, 123, 256, 1001, 2024)
DEFAULT_MASK_SEEDS = (101, 202, 303, 404, 505)
SUMMARY_METRICS = (
    "field_title_content_distance",
    "field_mask_to_named_centroid_distance",
    "qkv_joint_norm_l3_l2_ratio",
    "attention_out_norm_l3_l2_ratio",
    "ffn_joint_norm_l3_l2_ratio",
)


def _parse_ints(value: str, *, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must not be empty")
    return values


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = average
        start = end
    return ranks


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Spearman correlation requires equal lists with at least two values")
    rx, ry = _rank(x), _rank(y)
    mean_x, mean_y = statistics.fmean(rx), statistics.fmean(ry)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    denominator = math.sqrt(
        sum((a - mean_x) ** 2 for a in rx) * sum((b - mean_y) ** 2 for b in ry)
    )
    return numerator / denominator if denominator else float("nan")


def _read_final_outcomes(path: Path, seeds: tuple[int, ...]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            seed = int(raw["seed"])
            if {"delta_l2", "delta_l3"}.issubset(raw):
                delta_l2 = float(raw["delta_l2"])
                delta_l3 = float(raw["delta_l3"])
            else:
                delta_l2 = float(raw["saab_l2"]) - float(raw["baseline_l2"])
                delta_l3 = float(raw["saab_l3"]) - float(raw["baseline_l3"])
            rows[seed] = {
                "delta_l2": delta_l2,
                "delta_l3": delta_l3,
                "displacement_score": delta_l3 - delta_l2,
                "pattern": raw.get("pattern", ""),
            }
    missing = [seed for seed in seeds if seed not in rows]
    if missing:
        raise ValueError(f"final-outcomes CSV is missing seeds: {missing}")
    return {seed: rows[seed] for seed in seeds}


def _read_records(path: Path, count: int) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            arrays = [record.get(key) for key in ("input_ids", "field_ids", "attention_mask")]
            if any(not isinstance(value, list) for value in arrays):
                raise ValueError(f"{path}:{line_number} is missing prepared tensor lists")
            if len({len(value) for value in arrays}) != 1:
                raise ValueError(f"{path}:{line_number} has misaligned prepared tensors")
            records.append(record)
            if len(records) == count:
                break
    if len(records) != count:
        raise ValueError(f"{path} contains only {len(records)} records; expected {count}")
    return records


def _collate(records: list[dict[str, Any]], max_length: int):
    import torch

    length = min(max(len(row["input_ids"]) for row in records), max_length)
    input_ids = torch.zeros(len(records), length, dtype=torch.long)
    field_ids = torch.zeros(len(records), length, dtype=torch.long)
    attention_mask = torch.zeros(len(records), length, dtype=torch.bool)
    for row_index, record in enumerate(records):
        size = min(len(record["input_ids"]), length)
        input_ids[row_index, :size] = torch.tensor(record["input_ids"][:size])
        field_ids[row_index, :size] = torch.tensor(record["field_ids"][:size])
        attention_mask[row_index, :size] = torch.tensor(
            record["attention_mask"][:size], dtype=torch.bool
        )
    return input_ids, field_ids, attention_mask


def _state_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _joint_weight_norm(modules: Iterable[Any]) -> float:
    import torch

    total = sum(torch.sum(module.weight.detach().float().square()) for module in modules)
    return float(torch.sqrt(total))


def _initialization_summaries(model) -> tuple[dict[str, float], list[dict[str, float]]]:
    import torch

    layer_rows = []
    for index, layer in enumerate(model.encoder.layers):
        layer_rows.append(
            {
                "layer": index,
                "qkv_joint_norm": _joint_weight_norm(
                    [layer.attention.q_proj, layer.attention.k_proj, layer.attention.v_proj]
                ),
                "attention_out_norm": float(
                    torch.linalg.vector_norm(layer.attention.out_proj.weight.detach().float())
                ),
                "ffn_joint_norm": _joint_weight_norm([layer.ff[0], layer.ff[3]]),
            }
        )
    if len(layer_rows) != 4:
        raise ValueError("Initialization analysis requires the reported four-layer model")
    field = model.embeddings.field_embeddings.weight.detach().float()
    title, content, mask = field[4], field[3], field[5]
    named_centroid = (title + content) / 2.0
    compact = {
        "field_title_content_distance": float(torch.linalg.vector_norm(title - content)),
        "field_mask_to_named_centroid_distance": float(
            torch.linalg.vector_norm(mask - named_centroid)
        ),
        "qkv_joint_norm_l3_l2_ratio": (
            layer_rows[3]["qkv_joint_norm"] / layer_rows[2]["qkv_joint_norm"]
        ),
        "attention_out_norm_l3_l2_ratio": (
            layer_rows[3]["attention_out_norm"] / layer_rows[2]["attention_out_norm"]
        ),
        "ffn_joint_norm_l3_l2_ratio": (
            layer_rows[3]["ffn_joint_norm"] / layer_rows[2]["ffn_joint_norm"]
        ),
    }
    return compact, layer_rows


def _make_config(recipe: dict[str, Any], variant: str, vocab_size: int):
    from structformer.models import TransformerConfig

    data, model = recipe["data"], recipe["model_config"]
    return TransformerConfig(
        vocab_size=vocab_size,
        field_vocab_size=int(data["field_vocab_size"]),
        max_length=int(data["max_length"]),
        variant=variant,
        head_type="token",
        num_labels=int(data["num_labels"]),
        d_model=int(model["d_model"]),
        num_layers=int(model["num_layers"]),
        num_heads=int(model["num_heads"]),
        ff_dim=int(model["ff_dim"]),
        dropout=float(model["dropout"]),
        pad_token_id=int(data["pad_token_id"]),
        scale_embeddings=bool(model.get("scale_embeddings", False)),
        saab_field_weight=float(model.get("saab_field_weight", 1.0)),
    )


def _masked_probe_batches(blocks, mask_seeds, recipe):
    import torch

    from structformer.tasks.msm import mask_field_ids

    data = recipe["data"]
    probes = {}
    for mask_seed in mask_seeds:
        for block_index, batch in enumerate(blocks):
            generator = torch.Generator().manual_seed(mask_seed)
            masked, labels, positions = mask_field_ids(
                batch[1],
                batch[2],
                mask_field_id=int(data["mask_field_id"]),
                mask_probability=float(data["mask_probability"]),
                generator=generator,
            )
            probes[(mask_seed, block_index)] = (
                batch[0],
                masked,
                batch[2],
                labels,
                int(positions.sum()),
            )
    return probes


def _evaluate_initial_losses(model, probes, device, seed: int, variant: str):
    import torch

    from structformer.tasks.msm import msm_cross_entropy

    model.eval()
    rows = []
    with torch.inference_mode():
        for (mask_seed, block), probe in sorted(probes.items()):
            input_ids, field_ids, attention_mask, labels, masked_tokens = probe
            output = model(
                input_ids.to(device),
                field_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            loss = msm_cross_entropy(output.logits, labels.to(device))
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "mask_seed": mask_seed,
                    "block": block,
                    "n_examples": input_ids.shape[0],
                    "masked_tokens": masked_tokens,
                    "loss": float(loss.detach().cpu()),
                }
            )
    return rows


def _qkv_gradient_norms(model) -> list[float]:
    import torch

    values = []
    for layer in model.encoder.layers:
        total = torch.zeros((), device=next(model.parameters()).device)
        for projection in (layer.attention.q_proj, layer.attention.k_proj, layer.attention.v_proj):
            total = total + projection.weight.grad.detach().float().square().sum()
        values.append(float(total.sqrt().cpu()))
    return values


def _qkv_update_norms(model, before: list[list[Any]]) -> list[float]:
    import torch

    values = []
    for layer, saved in zip(model.encoder.layers, before):
        total = torch.zeros((), device=next(model.parameters()).device)
        projections = (layer.attention.q_proj, layer.attention.k_proj, layer.attention.v_proj)
        for projection, old in zip(projections, saved):
            total = total + (projection.weight.detach().float() - old).square().sum()
        values.append(float(total.sqrt().cpu()))
    return values


def _one_update_probe(model, probe, recipe, device) -> dict[str, float]:
    import torch

    from structformer.tasks.msm import msm_cross_entropy

    training = recipe["training"]
    input_ids, field_ids, attention_mask, labels, _ = probe
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]) * float(training.get("min_lr_ratio", 0.1)),
        weight_decay=float(training["weight_decay"]),
        betas=tuple(float(value) for value in training["betas"]),
    )
    before = [
        [
            projection.weight.detach().float().clone()
            for projection in (layer.attention.q_proj, layer.attention.k_proj, layer.attention.v_proj)
        ]
        for layer in model.encoder.layers
    ]
    optimizer.zero_grad(set_to_none=True)
    output = model(
        input_ids.to(device),
        field_ids.to(device),
        attention_mask=attention_mask.to(device),
    )
    loss = msm_cross_entropy(output.logits, labels.to(device))
    loss.backward()
    gradient_norms = _qkv_gradient_norms(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["grad_clip"]))
    optimizer.step()
    update_norms = _qkv_update_norms(model, before)
    return {
        "loss": float(loss.detach().cpu()),
        "qkv_gradient_ratio_l3_l2": gradient_norms[3] / gradient_norms[2],
        "qkv_adamw_update_ratio_l3_l2": update_norms[3] / update_norms[2],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(args: argparse.Namespace) -> Path:
    import torch
    import yaml

    from structformer.models import StructuredTransformerModel
    from structformer.utils.seed import seed_everything

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    recipe = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    outcomes = _read_final_outcomes(args.final_outcomes_csv, args.seeds)
    records = _read_records(args.validation_jsonl, args.probe_examples)
    block_size = args.probe_examples // args.probe_blocks
    blocks = [
        _collate(records[start : start + block_size], int(recipe["data"]["max_length"]))
        for start in range(0, args.probe_examples, block_size)
    ]
    probes = _masked_probe_batches(blocks, args.mask_seeds, recipe)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    hash_rows: list[dict[str, Any]] = []
    raw_loss_rows: list[dict[str, Any]] = []
    one_update_rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        print(f"initialization-sensitivity seed={seed} device={device}", flush=True)
        variant_hashes = {}
        for variant in ("baseline", "saab"):
            seed_everything(seed)
            model = StructuredTransformerModel(
                _make_config(recipe, variant, args.vocab_size)
            ).to(device)
            variant_hashes[variant] = _state_hash(model)
            if variant == "baseline":
                compact, per_layer = _initialization_summaries(model)
                summary_rows.append({"seed": seed, **compact, **outcomes[seed]})
                layer_rows.extend({"seed": seed, **row} for row in per_layer)
            raw_loss_rows.extend(
                _evaluate_initial_losses(model, probes, device, seed, variant)
            )
            if args.run_one_update:
                seed_everything(seed)
                result = _one_update_probe(
                    model,
                    probes[(args.mask_seeds[0], 0)],
                    recipe,
                    device,
                )
                one_update_rows.append({"seed": seed, "variant": variant, **result})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if variant_hashes["baseline"] != variant_hashes["saab"]:
            raise RuntimeError(f"Baseline/SAAB initial tensors differ at seed {seed}")
        hash_rows.append(
            {
                "seed": seed,
                "baseline_sha256": variant_hashes["baseline"],
                "saab_sha256": variant_hashes["saab"],
                "identical": True,
            }
        )

    displacement = [float(row["displacement_score"]) for row in summary_rows]
    correlation_rows = [
        {
            "analysis": "initial_parameter_summary",
            "metric": metric,
            "spearman_rho": _spearman(
                [float(row[metric]) for row in summary_rows], displacement
            ),
            "n_seeds": len(summary_rows),
        }
        for metric in SUMMARY_METRICS
    ]

    indexed = {
        (int(row["seed"]), str(row["variant"]), int(row["mask_seed"]), int(row["block"])): row
        for row in raw_loss_rows
    }
    paired_rows = []
    for seed in args.seeds:
        for mask_seed in args.mask_seeds:
            for block in range(args.probe_blocks):
                baseline = indexed[(seed, "baseline", mask_seed, block)]
                saab = indexed[(seed, "saab", mask_seed, block)]
                paired_rows.append(
                    {
                        "seed": seed,
                        "mask_seed": mask_seed,
                        "block": block,
                        "baseline": baseline["loss"],
                        "saab": saab["loss"],
                        "masked_tokens": baseline["masked_tokens"],
                        "loss_change": float(saab["loss"]) - float(baseline["loss"]),
                        "final_displacement_score": outcomes[seed]["displacement_score"],
                    }
                )
    by_seed_mask = []
    for seed in args.seeds:
        for mask_seed in args.mask_seeds:
            values = [
                float(row["loss_change"])
                for row in paired_rows
                if row["seed"] == seed and row["mask_seed"] == mask_seed
            ]
            by_seed_mask.append(
                {
                    "seed": seed,
                    "mask_seed": mask_seed,
                    "mean_loss_change": statistics.fmean(values),
                    "final_displacement_score": outcomes[seed]["displacement_score"],
                }
            )
    loss_summary = []
    for seed in args.seeds:
        values = [
            float(row["mean_loss_change"])
            for row in by_seed_mask
            if row["seed"] == seed
        ]
        loss_summary.append(
            {
                "seed": seed,
                "mean_initial_loss_change": statistics.fmean(values),
                "sd_across_masks": statistics.stdev(values),
                "minimum_loss_change": min(values),
                "maximum_loss_change": max(values),
                "positive_mask_fraction": sum(value > 0 for value in values) / len(values),
                "final_displacement_score": outcomes[seed]["displacement_score"],
            }
        )
    correlation_rows.append(
        {
            "analysis": "initial_loss_overall_mean",
            "metric": "saab_minus_baseline_loss",
            "spearman_rho": _spearman(
                [float(row["mean_initial_loss_change"]) for row in loss_summary],
                displacement,
            ),
            "n_seeds": len(args.seeds),
        }
    )
    for mask_seed in args.mask_seeds:
        correlation_rows.append(
            {
                "analysis": f"initial_loss_mask_{mask_seed}",
                "metric": "saab_minus_baseline_loss",
                "spearman_rho": _spearman(
                    [
                        float(next(row["mean_loss_change"] for row in by_seed_mask if row["seed"] == seed and row["mask_seed"] == mask_seed))
                        for seed in args.seeds
                    ],
                    displacement,
                ),
                "n_seeds": len(args.seeds),
            }
        )
    excluded_first = []
    for seed in args.seeds:
        excluded_first.append(
            statistics.fmean(
                float(row["loss_change"])
                for row in paired_rows
                if row["seed"] == seed and int(row["block"]) > 0
            )
        )
    correlation_rows.append(
        {
            "analysis": "initial_loss_excluding_first_64",
            "metric": "saab_minus_baseline_loss",
            "spearman_rho": _spearman(excluded_first, displacement),
            "n_seeds": len(args.seeds),
        }
    )
    for removed_seed in args.seeds:
        kept = [index for index, seed in enumerate(args.seeds) if seed != removed_seed]
        correlation_rows.append(
            {
                "analysis": f"initial_loss_leave_out_seed_{removed_seed}",
                "metric": "saab_minus_baseline_loss",
                "spearman_rho": _spearman(
                    [float(loss_summary[index]["mean_initial_loss_change"]) for index in kept],
                    [displacement[index] for index in kept],
                ),
                "n_seeds": len(kept),
            }
        )

    if one_update_rows:
        update_index = {
            (int(row["seed"]), str(row["variant"])): row for row in one_update_rows
        }
        paired_update_rows = []
        for seed in args.seeds:
            baseline, saab = update_index[(seed, "baseline")], update_index[(seed, "saab")]
            paired_update_rows.append(
                {
                    "seed": seed,
                    "qkv_gradient_ratio_change": (
                        float(saab["qkv_gradient_ratio_l3_l2"])
                        - float(baseline["qkv_gradient_ratio_l3_l2"])
                    ),
                    "qkv_adamw_update_ratio_change": (
                        float(saab["qkv_adamw_update_ratio_l3_l2"])
                        - float(baseline["qkv_adamw_update_ratio_l3_l2"])
                    ),
                    "final_displacement_score": outcomes[seed]["displacement_score"],
                }
            )
        _write_csv(args.out_dir / "one_update_paired.csv", paired_update_rows)
        for metric in ("qkv_gradient_ratio_change", "qkv_adamw_update_ratio_change"):
            correlation_rows.append(
                {
                    "analysis": "one_update_probe",
                    "metric": metric,
                    "spearman_rho": _spearman(
                        [float(row[metric]) for row in paired_update_rows], displacement
                    ),
                    "n_seeds": len(args.seeds),
                }
            )

    _write_csv(args.out_dir / "initialization_summary_by_seed.csv", summary_rows)
    _write_csv(args.out_dir / "layerwise_initial_norms.csv", layer_rows)
    _write_csv(args.out_dir / "reconstructed_state_hashes.csv", hash_rows)
    _write_csv(args.out_dir / "robust_initial_loss_raw.csv", raw_loss_rows)
    _write_csv(args.out_dir / "robust_initial_loss_paired.csv", paired_rows)
    _write_csv(args.out_dir / "robust_initial_loss_by_seed_mask.csv", by_seed_mask)
    _write_csv(args.out_dir / "robust_initial_loss_summary.csv", loss_summary)
    _write_csv(args.out_dir / "initialization_correlations.csv", correlation_rows)
    if one_update_rows:
        _write_csv(args.out_dir / "one_update_raw.csv", one_update_rows)

    provenance = {
        "analysis": "exploratory_initialization_sensitivity",
        "git_commit": _git_commit(),
        "config": str(args.config),
        "validation_jsonl": str(args.validation_jsonl),
        "final_outcomes_csv": str(args.final_outcomes_csv),
        "seeds": list(args.seeds),
        "mask_seeds": list(args.mask_seeds),
        "probe_examples": args.probe_examples,
        "probe_blocks": args.probe_blocks,
        "dropout_disabled_for_initial_loss": True,
        "one_update_probe": args.run_one_update,
        "one_update_batch": "first block under the first deterministic mask seed",
        "vocab_size": args.vocab_size,
        "device": str(device),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "interpretation_limit": (
            "Exploratory descriptive analysis over the same eight final outcomes; "
            "not a causal mechanism or validated predictor for new initializations."
        ),
    }
    (args.out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"initialization-sensitivity complete output={args.out_dir}", flush=True)
    return args.out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("dbpedia/configs/msm_dbpedia_full_recipe.yaml"),
    )
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--final-outcomes-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument(
        "--mask-seeds", default=",".join(str(seed) for seed in DEFAULT_MASK_SEEDS)
    )
    parser.add_argument("--probe-examples", type=int, default=256)
    parser.add_argument("--probe-blocks", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=30_000)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--skip-one-update", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.seeds = _parse_ints(args.seeds, name="seeds")
    args.mask_seeds = _parse_ints(args.mask_seeds, name="mask-seeds")
    if args.probe_examples <= 0 or args.probe_blocks <= 0:
        raise ValueError("probe sizes must be positive")
    if args.probe_examples % args.probe_blocks:
        raise ValueError("probe-examples must be divisible by probe-blocks")
    if args.probe_examples // args.probe_blocks != 64:
        raise ValueError("the reported analysis requires four 64-example blocks")
    if args.vocab_size <= 0:
        raise ValueError("vocab-size must be positive")
    args.run_one_update = not args.skip_one_update
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
