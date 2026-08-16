#!/usr/bin/env python3
"""Export a reproducible individual-token attention example for DBpedia.

The diagnostic evaluates one output-independently selected record from the
predeclared untouched-test reservoir sample. It compares the exact paired
seed-1001 Baseline and SAAB checkpoints, averages attention over every head,
and exports both numerical matrices and publication-candidate figures. The
result is qualitative and must not be used as inferential evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_paired_sfm_length import (
    DatasetSpec,
    _field_ids,
    _load_checkpoint,
    _rebuild_model,
    _select_records,
    _sha256,
    _verify_checkpoint,
)


EXPECTED_SOURCE_RECORDS = 70_000
EXPECTED_TEST_SPLIT = "test"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_field_names(path: Path, mask_field_id: int) -> dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = {
        int(field_id): str(name)
        for name, field_id in payload["field_vocab"].items()
        if int(field_id) > 2 and int(field_id) != mask_field_id
    }
    if not names:
        raise ValueError(f"No named field IDs found in {path}")
    return dict(sorted(names.items()))


def _valid_positions(record: dict[str, Any], max_length: int) -> list[int]:
    arrays = [record.get(name) for name in ("input_ids", "field_ids", "attention_mask")]
    if any(not isinstance(value, list) for value in arrays):
        raise ValueError("record is missing prepared tensor lists")
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("record tensor lists are not aligned")
    tokens = record.get("tokens")
    if not isinstance(tokens, list) or len(tokens) != len(record["input_ids"]):
        raise ValueError("record tokens are missing or misaligned")
    return [
        index
        for index, is_valid in enumerate(record["attention_mask"][:max_length])
        if bool(is_valid)
    ]


def _eligibility(
    record: dict[str, Any],
    named_field_ids: list[int],
    *,
    min_length: int,
    max_length: int,
    min_tokens_per_field: int,
    evaluation_max_length: int,
) -> tuple[bool, dict[str, Any]]:
    positions = _valid_positions(record, evaluation_max_length)
    counts = {
        field_id: sum(record["field_ids"][index] == field_id for index in positions)
        for field_id in named_field_ids
    }
    eligible = (
        min_length <= len(positions) <= max_length
        and all(count >= min_tokens_per_field for count in counts.values())
        and all(record["field_ids"][index] in named_field_ids for index in positions)
    )
    return eligible, {
        "valid_token_count": len(positions),
        "named_field_token_counts": {str(key): value for key, value in counts.items()},
    }


def _select_candidate(
    test_jsonl: Path,
    named_field_ids: list[int],
    *,
    sample_size: int,
    analysis_seed: int,
    min_length: int,
    max_length: int,
    min_tokens_per_field: int,
    evaluation_max_length: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected, _, sample_indices, source_records = _select_records(
        test_jsonl,
        primary_size=1,
        length_sample_size=sample_size,
        sample_seed=analysis_seed,
    )
    if source_records != EXPECTED_SOURCE_RECORDS:
        raise ValueError(
            f"expected {EXPECTED_SOURCE_RECORDS} source records, found {source_records}"
        )
    candidates = sorted(
        (record for record in selected if int(record["source_index"]) in sample_indices),
        key=lambda record: int(record["source_index"]),
    )
    for record in candidates:
        eligible, details = _eligibility(
            record,
            named_field_ids,
            min_length=min_length,
            max_length=max_length,
            min_tokens_per_field=min_tokens_per_field,
            evaluation_max_length=evaluation_max_length,
        )
        if eligible:
            rule = {
                "reservoir_sample_size": sample_size,
                "reservoir_seed": analysis_seed,
                "ordering": "lowest source index among eligible reservoir records",
                "minimum_valid_tokens": min_length,
                "maximum_valid_tokens": max_length,
                "minimum_tokens_per_named_field": min_tokens_per_field,
                "model_outputs_used_for_selection": False,
            }
            return record, {**details, "selection_rule": rule}
    raise ValueError("No record satisfies the predeclared readability criteria")


def _checkpoint_pair(
    baseline_path: Path,
    saab_path: Path,
    *,
    expected_baseline_sha256: str | None,
    expected_saab_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    observed_hashes = {
        "baseline": _sha256(baseline_path),
        "saab": _sha256(saab_path),
    }
    expected_hashes = {
        "baseline": expected_baseline_sha256,
        "saab": expected_saab_sha256,
    }
    for variant, expected in expected_hashes.items():
        if expected and observed_hashes[variant] != expected:
            raise ValueError(
                f"{variant} checkpoint SHA-256 mismatch: {observed_hashes[variant]}"
            )

    spec = DatasetSpec(
        "DBpedia-test-token-map",
        baseline_path,
        saab_path,
        Path("unused.jsonl"),
        Path("unused-field-vocab.json"),
        500,
        256,
        5,
        6,
    )
    baseline = _load_checkpoint(baseline_path)
    saab = _load_checkpoint(saab_path)
    _verify_checkpoint(baseline, spec, "baseline")
    _verify_checkpoint(saab, spec, "saab")
    saab_model_config = saab["config"]["model_config"]
    if float(saab_model_config.get("saab_field_weight", 1.0)) != 1.0:
        raise ValueError("SAAB checkpoint does not use the fixed paper weight w_f=1.0")
    if bool(saab_model_config.get("saab_shuffle_bias", False)):
        raise ValueError("SAAB checkpoint unexpectedly enables shuffled bias")
    layer_mask = tuple(float(value) for value in saab_model_config.get("saab_layer_mask", ()))
    if layer_mask and layer_mask != (1.0, 1.0, 1.0, 1.0):
        raise ValueError(f"SAAB checkpoint uses an unexpected layer mask: {layer_mask}")
    identities = {
        variant: {
            "sha256": observed_hashes[variant],
            "seed": checkpoint["config"]["seed"],
            "step": checkpoint["step"],
            "variant": checkpoint["config"]["model"],
            "max_length": checkpoint["config"]["data"]["max_length"],
            "num_layers": checkpoint["config"]["model_config"]["num_layers"],
            "num_heads": checkpoint["config"]["model_config"]["num_heads"],
        }
        for variant, checkpoint in (("baseline", baseline), ("saab", saab))
    }
    return baseline, saab, identities


def _choose_device(requested: str, allow_cpu: bool):
    import torch

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Select a Colab GPU runtime.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable.")
        return torch.device("mps")
    if requested == "cpu":
        if not allow_cpu:
            raise RuntimeError("CPU requires the explicit --allow-cpu option.")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError("No CUDA or MPS device is available; CPU was not explicitly allowed.")


def _evaluate_attention(
    checkpoint: dict[str, Any],
    record: dict[str, Any],
    *,
    layers: list[int],
    evaluation_max_length: int,
    device,
) -> dict[int, list[list[float]]]:
    import torch

    model = _rebuild_model(checkpoint, device)
    length = min(len(record["input_ids"]), evaluation_max_length)
    input_ids = torch.tensor(
        record["input_ids"][:length], dtype=torch.long, device=device
    ).unsqueeze(0)
    field_ids = torch.tensor(
        record["field_ids"][:length], dtype=torch.long, device=device
    ).unsqueeze(0)
    attention_mask = torch.tensor(
        record["attention_mask"][:length], dtype=torch.bool, device=device
    ).unsqueeze(0)
    with torch.inference_mode():
        output = model(
            input_ids,
            field_ids,
            attention_mask=attention_mask,
            need_weights=True,
        )
    attentions = output.attentions
    if not attentions:
        raise RuntimeError("model did not return attention matrices")
    invalid_layers = [layer for layer in layers if layer < 0 or layer >= len(attentions)]
    if invalid_layers:
        raise ValueError(f"invalid layer indices: {invalid_layers}")
    positions = torch.nonzero(attention_mask[0], as_tuple=False).flatten()
    matrices: dict[int, list[list[float]]] = {}
    for layer in layers:
        head_mean = attentions[layer][0].mean(dim=0)
        valid_matrix = head_mean.index_select(0, positions).index_select(1, positions)
        row_sums = valid_matrix.sum(dim=-1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5):
            raise RuntimeError(f"L{layer} valid-key attention rows do not sum to one")
        matrices[layer] = valid_matrix.detach().cpu().tolist()
    del output, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return matrices


def _same_field_mass(matrix: list[list[float]], field_ids: list[int]) -> float:
    if len(matrix) != len(field_ids) or any(len(row) != len(field_ids) for row in matrix):
        raise ValueError("attention matrix and field IDs are misaligned")
    masses = [
        sum(value for value, key_field in zip(row, field_ids) if key_field == query_field)
        for row, query_field in zip(matrix, field_ids)
    ]
    return sum(masses) / len(masses)


def _field_to_field(
    matrix: list[list[float]],
    field_ids: list[int],
    field_names: dict[int, str],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for source_id, source_name in field_names.items():
        source_positions = [i for i, field_id in enumerate(field_ids) if field_id == source_id]
        if not source_positions:
            continue
        result[source_name] = {}
        for destination_id, destination_name in field_names.items():
            destination_positions = [
                i for i, field_id in enumerate(field_ids) if field_id == destination_id
            ]
            masses = [
                sum(matrix[source][destination] for destination in destination_positions)
                for source in source_positions
            ]
            result[source_name][destination_name] = sum(masses) / len(masses)
    return result


def _field_boundaries(field_ids: list[int]) -> list[int]:
    return [
        index
        for index in range(1, len(field_ids))
        if field_ids[index] != field_ids[index - 1]
    ]


def _top_changes(
    baseline: list[list[float]],
    saab: list[list[float]],
    tokens: list[str],
    field_ids: list[int],
    field_names: dict[int, str],
    *,
    layer: int,
    limit: int,
) -> list[dict[str, Any]]:
    rows = []
    for query in range(len(tokens)):
        for key in range(len(tokens)):
            delta = saab[query][key] - baseline[query][key]
            rows.append(
                {
                    "layer": layer,
                    "query_position": query,
                    "query_token": tokens[query],
                    "query_field": field_names[field_ids[query]],
                    "key_position": key,
                    "key_token": tokens[key],
                    "key_field": field_names[field_ids[key]],
                    "baseline_attention": baseline[query][key],
                    "saab_attention": saab[query][key],
                    "saab_minus_baseline": delta,
                    "absolute_change": abs(delta),
                }
            )
    return sorted(
        rows,
        key=lambda row: (-row["absolute_change"], row["query_position"], row["key_position"]),
    )[:limit]


def _plot_candidate(
    path_stem: Path,
    matrices: dict[str, dict[int, list[list[float]]]],
    tokens: list[str],
    field_ids: list[int],
    field_names: dict[int, str],
    layers: list[int],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to render the candidate figure. Install it "
            "manually in the active environment; this script does not install packages."
        ) from exc

    raw_max = max(
        value
        for variant in ("baseline", "saab")
        for layer in layers
        for row in matrices[variant][layer]
        for value in row
    )
    differences = {
        layer: [
            [
                matrices["saab"][layer][query][key]
                - matrices["baseline"][layer][query][key]
                for key in range(len(tokens))
            ]
            for query in range(len(tokens))
        ]
        for layer in layers
    }
    difference_max = max(
        abs(value) for layer in layers for row in differences[layer] for value in row
    )
    difference_max = max(difference_max, 1e-12)
    boundaries = _field_boundaries(field_ids)
    labels = [f"{position}:{token}" for position, token in enumerate(tokens)]

    figure, axes = plt.subplots(
        len(layers), 3, figsize=(18, 5.9 * len(layers)), squeeze=False
    )
    raw_image = None
    difference_image = None
    for row_index, layer in enumerate(layers):
        panels = [
            ("Baseline", matrices["baseline"][layer], "YlOrBr", None),
            ("SAAB", matrices["saab"][layer], "YlOrBr", None),
            (
                "SAAB − Baseline",
                differences[layer],
                "BrBG",
                TwoSlopeNorm(vmin=-difference_max, vcenter=0.0, vmax=difference_max),
            ),
        ]
        for column, (title, values, color_map, norm) in enumerate(panels):
            axis = axes[row_index][column]
            if norm is None:
                image = axis.imshow(
                    values, cmap=color_map, vmin=0.0, vmax=raw_max, interpolation="nearest"
                )
                raw_image = image
            else:
                image = axis.imshow(values, cmap=color_map, norm=norm, interpolation="nearest")
                difference_image = image
            axis.set_title(f"L{layer}: {title}", fontsize=12, fontweight="bold")
            axis.set_xticks(range(len(tokens)))
            axis.set_yticks(range(len(tokens)))
            axis.set_xticklabels(labels, rotation=90, fontsize=6)
            axis.set_yticklabels(labels, fontsize=6)
            axis.set_xlabel("Key token", fontsize=10)
            axis.set_ylabel("Query token", fontsize=10)
            for boundary in boundaries:
                axis.axvline(boundary - 0.5, color="#007C83", linewidth=1.2)
                axis.axhline(boundary - 0.5, color="#007C83", linewidth=1.2)
    assert raw_image is not None and difference_image is not None
    figure.colorbar(
        raw_image,
        ax=[axes[row][column] for row in range(len(layers)) for column in (0, 1)],
        fraction=0.018,
        pad=0.015,
        label="Mean attention weight across all heads",
    )
    figure.colorbar(
        difference_image,
        ax=[axes[row][2] for row in range(len(layers))],
        fraction=0.035,
        pad=0.015,
        label="SAAB − Baseline attention weight",
    )
    segment_text = ", ".join(
        f"{name}: {sum(field_id == fid for field_id in field_ids)} tokens"
        for fid, name in field_names.items()
    )
    figure.suptitle(
        "Individual-token attention for one output-independently selected DBpedia test record\n"
        + segment_text,
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    figure.subplots_adjust(left=0.075, right=0.91, bottom=0.13, top=0.92, wspace=0.42, hspace=0.36)
    figure.savefig(path_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    import torch

    required = [
        args.baseline_checkpoint,
        args.saab_checkpoint,
        args.test_jsonl,
        args.test_manifest,
        args.field_vocab_json,
    ]
    missing = [path for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(str(path) for path in missing))
    manifest = json.loads(args.test_manifest.read_text(encoding="utf-8"))
    if manifest.get("split") != EXPECTED_TEST_SPLIT:
        raise ValueError("manifest does not describe the official test split")
    if int(manifest.get("records", -1)) != EXPECTED_SOURCE_RECORDS:
        raise ValueError("manifest does not describe 70,000 source records")
    test_hash = _sha256(args.test_jsonl)
    if manifest.get("output_jsonl_sha256") != test_hash:
        raise ValueError("test JSONL hash does not match its preparation manifest")

    device = _choose_device(args.device, args.allow_cpu)
    print(f"device={device}", flush=True)
    print("Loading and verifying exact checkpoint pair...", flush=True)
    baseline_checkpoint, saab_checkpoint, checkpoint_identities = _checkpoint_pair(
        args.baseline_checkpoint,
        args.saab_checkpoint,
        expected_baseline_sha256=args.expected_baseline_sha256,
        expected_saab_sha256=args.expected_saab_sha256,
    )
    mask_field_id = int(baseline_checkpoint["config"]["data"]["mask_field_id"])
    named_field_ids = _field_ids(args.field_vocab_json, mask_field_id)
    field_names = _load_field_names(args.field_vocab_json, mask_field_id)
    if named_field_ids != list(field_names):
        raise ValueError("field-vocabulary name and ID reconstruction disagree")

    print("Selecting the first eligible record without inspecting model outputs...", flush=True)
    record, selection = _select_candidate(
        args.test_jsonl,
        named_field_ids,
        sample_size=args.sample_size,
        analysis_seed=args.analysis_seed,
        min_length=args.min_length,
        max_length=args.max_length,
        min_tokens_per_field=args.min_tokens_per_field,
        evaluation_max_length=256,
    )
    valid_positions = _valid_positions(record, 256)
    tokens = [str(record["tokens"][index]) for index in valid_positions]
    field_ids = [int(record["field_ids"][index]) for index in valid_positions]
    print(
        f"selected source_index={record['source_index']} row_id={record.get('row_id')} "
        f"valid_tokens={len(tokens)}",
        flush=True,
    )

    print("Evaluating Baseline attention...", flush=True)
    baseline_matrices = _evaluate_attention(
        baseline_checkpoint,
        record,
        layers=args.layers,
        evaluation_max_length=256,
        device=device,
    )
    del baseline_checkpoint
    print("Evaluating SAAB attention...", flush=True)
    saab_matrices = _evaluate_attention(
        saab_checkpoint,
        record,
        layers=args.layers,
        evaluation_max_length=256,
        device=device,
    )
    del saab_checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()

    matrices = {"baseline": baseline_matrices, "saab": saab_matrices}
    summary_rows: list[dict[str, Any]] = []
    top_change_rows: list[dict[str, Any]] = []
    layer_summaries: dict[str, Any] = {}
    for layer in args.layers:
        baseline_sfm = _same_field_mass(baseline_matrices[layer], field_ids)
        saab_sfm = _same_field_mass(saab_matrices[layer], field_ids)
        summary_rows.append(
            {
                "layer": layer,
                "baseline_sfm": baseline_sfm,
                "saab_sfm": saab_sfm,
                "saab_minus_baseline_sfm": saab_sfm - baseline_sfm,
                "mean_absolute_token_pair_change": sum(
                    abs(saab_matrices[layer][query][key] - baseline_matrices[layer][query][key])
                    for query in range(len(tokens))
                    for key in range(len(tokens))
                )
                / (len(tokens) ** 2),
                "maximum_absolute_token_pair_change": max(
                    abs(saab_matrices[layer][query][key] - baseline_matrices[layer][query][key])
                    for query in range(len(tokens))
                    for key in range(len(tokens))
                ),
            }
        )
        top_change_rows.extend(
            _top_changes(
                baseline_matrices[layer],
                saab_matrices[layer],
                tokens,
                field_ids,
                field_names,
                layer=layer,
                limit=args.top_changes,
            )
        )
        layer_summaries[str(layer)] = {
            "baseline_field_to_field": _field_to_field(
                baseline_matrices[layer], field_ids, field_names
            ),
            "saab_field_to_field": _field_to_field(
                saab_matrices[layer], field_ids, field_names
            ),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    token_rows = [
        {
            "display_position": display_position,
            "source_position": source_position,
            "token": token,
            "field_id": field_id,
            "field_name": field_names[field_id],
        }
        for display_position, (source_position, token, field_id) in enumerate(
            zip(valid_positions, tokens, field_ids)
        )
    ]
    _write_csv(args.out_dir / "selected_record_tokens.csv", token_rows)
    _write_csv(args.out_dir / "example_summary.csv", summary_rows)
    _write_csv(args.out_dir / "largest_token_pair_changes.csv", top_change_rows)

    matrix_payload = {
        "axis_semantics": "rows are query tokens; columns are key tokens",
        "head_aggregation": "arithmetic mean over all six attention heads",
        "tokens": tokens,
        "field_ids": field_ids,
        "field_names": {str(key): value for key, value in field_names.items()},
        "layers": {
            str(layer): {
                "baseline": baseline_matrices[layer],
                "saab": saab_matrices[layer],
                "saab_minus_baseline": [
                    [
                        saab_matrices[layer][query][key]
                        - baseline_matrices[layer][query][key]
                        for key in range(len(tokens))
                    ]
                    for query in range(len(tokens))
                ],
            }
            for layer in args.layers
        },
    }
    (args.out_dir / "attention_matrices.json").write_text(
        json.dumps(matrix_payload, indent=2) + "\n", encoding="utf-8"
    )

    selection_payload = {
        "scope": "qualitative individual-token attention candidate",
        "source_split": "official untouched DBpedia test split",
        "source_records": EXPECTED_SOURCE_RECORDS,
        "test_jsonl_sha256": test_hash,
        "source_index": int(record["source_index"]),
        "row_id": record.get("row_id"),
        "selection": selection,
        "checkpoint_identities": checkpoint_identities,
        "evaluation_max_length": 256,
        "layers": args.layers,
        "head_aggregation": "mean over all heads",
        "code_commit": _git_commit(),
        "interpretation_boundary": (
            "This output-independent single-record visualization is qualitative. "
            "It illustrates token-level routing for one fixed model pair and does "
            "not provide inferential or across-seed evidence."
        ),
    }
    (args.out_dir / "selection_and_provenance.json").write_text(
        json.dumps(selection_payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "layer_summaries.json").write_text(
        json.dumps(layer_summaries, indent=2) + "\n", encoding="utf-8"
    )

    print("Rendering publication-candidate figure...", flush=True)
    _plot_candidate(
        args.out_dir / "individual_token_attention_candidate",
        matrices,
        tokens,
        field_ids,
        field_names,
        args.layers,
    )
    environment = {
        "device": device.type,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "python": sys.version,
        "code_commit": _git_commit(),
    }
    (args.out_dir / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved individual-token diagnostic to {args.out_dir}", flush=True)
    return args.out_dir


def _parse_layers(value: str) -> list[int]:
    layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not layers or len(layers) != len(set(layers)):
        raise argparse.ArgumentTypeError("layers must be a non-empty comma-separated set")
    return layers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--saab-checkpoint", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--field-vocab-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-baseline-sha256", default=None)
    parser.add_argument("--expected-saab-sha256", default=None)
    parser.add_argument("--sample-size", type=int, default=2336)
    parser.add_argument("--analysis-seed", type=int, default=1001)
    parser.add_argument("--min-length", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=30)
    parser.add_argument("--min-tokens-per-field", type=int, default=4)
    parser.add_argument("--layers", type=_parse_layers, default=[2, 3])
    parser.add_argument("--top-changes", type=int, default=20)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if args.min_length <= 0 or args.max_length < args.min_length:
        parser.error("invalid length bounds")
    if args.min_tokens_per_field <= 0 or args.top_changes <= 0:
        parser.error("token-count settings must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
