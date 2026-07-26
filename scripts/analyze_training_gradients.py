#!/usr/bin/env python3
"""Summarize per-step MSM attention-gradient logs from paired training runs.

The training loop records the joint Q/K/V weight-gradient norm for every
Transformer layer after gradient accumulation and before global clipping.
This script compares the final two layers and computes the trailing-window
summaries used in the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_MODELS = ("baseline", "saab")
DEFAULT_CHECKPOINT_STEPS = (1, 10, 50, 100, 200, 300, 500)
EXPECTED_OBJECTIVE = "msm_cross_entropy"
EXPECTED_MEASUREMENT_POINT = "after_gradient_accumulation_before_global_clipping"


def _load_gradient_rows(
    path: Path,
    *,
    expected_model: str,
    expected_seed: int,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Gradient log not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("model") != expected_model:
                raise ValueError(
                    f"{path}:{line_number} has model={row.get('model')!r}; "
                    f"expected {expected_model!r}"
                )
            if int(row.get("seed")) != expected_seed:
                raise ValueError(
                    f"{path}:{line_number} has seed={row.get('seed')!r}; "
                    f"expected {expected_seed}"
                )
            if row.get("objective") != EXPECTED_OBJECTIVE:
                raise ValueError(
                    f"{path}:{line_number} has objective={row.get('objective')!r}; "
                    f"expected {EXPECTED_OBJECTIVE!r}"
                )
            if row.get("measurement_point") != EXPECTED_MEASUREMENT_POINT:
                raise ValueError(
                    f"{path}:{line_number} has measurement_point="
                    f"{row.get('measurement_point')!r}; "
                    f"expected {EXPECTED_MEASUREMENT_POINT!r}"
                )
            norms = row.get("qkv_grad_norm_per_layer")
            if not isinstance(norms, list) or len(norms) < 2:
                raise ValueError(
                    f"{path}:{line_number} does not contain at least two layer norms"
                )
            ratio = row.get("grad_norm_ratio_last_penultimate")
            if ratio is None:
                raise ValueError(f"{path}:{line_number} does not contain a gradient ratio")
            rows.append(row)

    rows.sort(key=lambda row: int(row["step"]))
    steps = [int(row["step"]) for row in rows]
    if len(steps) != len(set(steps)):
        raise ValueError(f"Duplicate steps found in {path}")
    return rows


def _trailing_means(values: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("rolling window must be positive")
    means: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        means.append(statistics.fmean(values[start : index + 1]))
    return means


def _checkpoint_summary(
    rows: list[dict[str, Any]],
    *,
    checkpoint_steps: tuple[int, ...],
    window: int,
) -> list[dict[str, Any]]:
    summarized: list[dict[str, Any]] = []
    for checkpoint in checkpoint_steps:
        eligible = [row for row in rows if int(row["step"]) <= checkpoint]
        if not eligible or int(eligible[-1]["step"]) != checkpoint:
            continue
        selected = eligible[-window:]
        ratios = [
            float(row["grad_norm_ratio_last_penultimate"]) for row in selected
        ]
        penultimate = [
            float(row["qkv_grad_norm_per_layer"][-2]) for row in selected
        ]
        last = [float(row["qkv_grad_norm_per_layer"][-1]) for row in selected]
        summarized.append(
            {
                "checkpoint_step": checkpoint,
                "window_start_step": int(selected[0]["step"]),
                "window_end_step": int(selected[-1]["step"]),
                "n_steps": len(selected),
                "mean_ratio_last_penultimate": statistics.fmean(ratios),
                "sd_ratio_last_penultimate": (
                    statistics.stdev(ratios) if len(ratios) > 1 else 0.0
                ),
                "mean_penultimate_qkv_grad_norm": statistics.fmean(penultimate),
                "mean_last_qkv_grad_norm": statistics.fmean(last),
            }
        )
    return summarized


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _write_checkpoint_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_results(
    by_model: dict[str, list[dict[str, Any]]],
    *,
    rolling_window: int,
    out_dir: Path,
) -> None:
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required to create gradient figures. "
            "Install the project runtime dependencies manually."
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"baseline": "#3b6fb6", "saab": "#b33b45"}

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for model, rows in by_model.items():
        steps = [int(row["step"]) for row in rows]
        ratios = [
            float(row["grad_norm_ratio_last_penultimate"]) for row in rows
        ]
        color = colors.get(model)
        ax.plot(steps, ratios, color=color, alpha=0.18, linewidth=0.8)
        ax.plot(
            steps,
            _trailing_means(ratios, rolling_window),
            color=color,
            linewidth=2.2,
            label=f"{model.capitalize()} ({rolling_window}-step mean)",
        )
    ax.axhline(
        1.0,
        color="#777777",
        linestyle="--",
        linewidth=1.2,
        label="Equal gradient norm",
    )
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("L3/L2 MSM loss-gradient norm ratio")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "training_grad_ratio_L3_L2.png", dpi=180)
    fig.savefig(out_dir / "training_grad_ratio_L3_L2.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(by_model), figsize=(10.6, 4.4), sharey=False)
    if len(by_model) == 1:
        axes = [axes]
    for ax, (model, rows) in zip(axes, by_model.items()):
        steps = [int(row["step"]) for row in rows]
        penultimate = [
            float(row["qkv_grad_norm_per_layer"][-2]) for row in rows
        ]
        last = [float(row["qkv_grad_norm_per_layer"][-1]) for row in rows]
        ax.plot(
            steps,
            _trailing_means(penultimate, rolling_window),
            linewidth=2.0,
            label="L2",
        )
        ax.plot(
            steps,
            _trailing_means(last, rolling_window),
            linewidth=2.0,
            label="L3",
        )
        ax.set_title(model.capitalize())
        ax.set_xlabel("Optimization step")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Joint Q/K/V MSM loss-gradient norm")
    fig.tight_layout()
    fig.savefig(out_dir / "training_grad_norm_L2_L3.png", dpi=180)
    fig.savefig(out_dir / "training_grad_norm_L2_L3.pdf")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze per-step MSM layer-gradient logs."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Root containing {model}_seed{seed}/layer_gradients/metrics.jsonl.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--models", default="baseline,saab")
    parser.add_argument("--rolling-window", type=int, default=10)
    parser.add_argument(
        "--checkpoint-steps",
        default="1,10,50,100,200,300,500",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    models = tuple(value.strip() for value in args.models.split(",") if value.strip())
    checkpoint_steps = tuple(
        int(value.strip())
        for value in args.checkpoint_steps.split(",")
        if value.strip()
    )
    if not models:
        raise ValueError("At least one model is required")
    if args.rolling_window <= 0:
        raise ValueError("rolling window must be positive")

    by_model: dict[str, list[dict[str, Any]]] = {}
    for model in models:
        path = (
            args.run_root
            / f"{model}_seed{args.seed}"
            / "layer_gradients"
            / "metrics.jsonl"
        )
        rows = _load_gradient_rows(
            path,
            expected_model=model,
            expected_seed=args.seed,
        )
        if not rows:
            raise ValueError(f"No gradient rows found in {path}")
        by_model[model] = rows

    reference_steps = [int(row["step"]) for row in next(iter(by_model.values()))]
    for model, rows in by_model.items():
        model_steps = [int(row["step"]) for row in rows]
        if model_steps != reference_steps:
            raise ValueError(
                f"Step coverage for {model} does not match the other runs"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_rows: list[dict[str, Any]] = []
    result_by_model: dict[str, Any] = {}
    for model, rows in by_model.items():
        ratios = [
            float(row["grad_norm_ratio_last_penultimate"]) for row in rows
        ]
        summaries = _checkpoint_summary(
            rows,
            checkpoint_steps=checkpoint_steps,
            window=args.rolling_window,
        )
        for summary in summaries:
            checkpoint_rows.append({"model": model, "seed": args.seed, **summary})
        result_by_model[model] = {
            "steps": [int(row["step"]) for row in rows],
            "ratio_last_penultimate": ratios,
            "rolling_mean_ratio": _trailing_means(ratios, args.rolling_window),
            "checkpoint_summaries": summaries,
        }

    result = {
        "seed": args.seed,
        "models": list(models),
        "git_commit": _git_commit(),
        "objective": "masked_structure_modeling_cross_entropy",
        "input": "the masked MSM training batches used for optimization",
        "measurement_point": EXPECTED_MEASUREMENT_POINT,
        "layer_metric": "joint L2 norm over Q/K/V projection weight gradients",
        "ratio": "last_layer_norm / penultimate_layer_norm",
        "rolling_window_steps": args.rolling_window,
        "checkpoint_steps": list(checkpoint_steps),
        "interpretation_limit": (
            "Gradient norms measure relative signal magnitude, not gradient direction "
            "or an AdamW parameter-update magnitude."
        ),
        "by_model": result_by_model,
    }
    with (args.out_dir / "training_gradient_diagnostics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    _write_checkpoint_csv(
        args.out_dir / "gradient_checkpoint_summary.csv",
        checkpoint_rows,
    )
    _plot_results(
        by_model,
        rolling_window=args.rolling_window,
        out_dir=args.out_dir,
    )

    print(f"Read {len(reference_steps)} training-gradient steps for: {', '.join(models)}")
    print(f"Saved gradient analysis to: {args.out_dir}")
    for row in checkpoint_rows:
        print(
            f"  {row['model']:<8} step={row['checkpoint_step']:>3} "
            f"window={row['window_start_step']:>3}-{row['window_end_step']:>3} "
            f"mean_ratio={row['mean_ratio_last_penultimate']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
