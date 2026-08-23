"""Time-series attention diagnostics across training steps.

Loads step_XXXX.pt diagnostic checkpoints saved during training and
computes structural-attention routing metrics at each step for the paired
Baseline and SAAB variants.

This answers the question: does SAAB's last-layer routing concentration
exist from step 1 (structural property of the bias), or does it emerge
through training?

Metrics computed at each step:
  - Same-field attention mass per layer
  - Field-to-field attention matrix (title/content) per layer
  - Attention entropy per layer

Output: timeseries_diagnostics.json indexed by step, then model.

Usage:
    PYTHONPATH=src python scripts/diag_timeseries.py \\
        --run-root runs/gradients \\
        --seed 1001 \\
        --val-jsonl /path/to/processed/dbpedia/val.jsonl \\
        --n-examples 32 \\
        --out-dir outputs/diagnostics/timeseries

    # Synthetic fallback (no val.jsonl needed):
    PYTHONPATH=src python scripts/diag_timeseries.py \\
        --run-root runs/gradients \\
        --seed 1001 \\
        --synthetic
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch

# Reuse metric helpers from diag_attention
sys.path.insert(0, str(Path(__file__).parent))
from diag_attention import (
    _attention_entropy,
    _detect_named_fields,
    _field_to_field_matrix,
    _load_checkpoint,
    _load_real_examples,
    _make_synthetic_batch,
    _rebuild_model,
    _same_field_mass,
    _valid_mask,
)

MODELS = ["baseline", "saab"]


def _find_diagnostic_checkpoints(run_root: Path, seed: int) -> dict[str, dict[int, Path]]:
    """Return {model: {step: path}} for all step_XXXX.pt files found."""
    found: dict[str, dict[int, Path]] = {m: {} for m in MODELS}
    for model in MODELS:
        ckpt_dir = run_root / f"{model}_seed{seed}" / "checkpoints"
        if not ckpt_dir.exists():
            print(f"  WARNING: checkpoint dir not found: {ckpt_dir}")
            continue
        for p in sorted(ckpt_dir.glob("step_*.pt")):
            m = re.match(r"step_(\d+)\.pt", p.name)
            if m:
                found[model][int(m.group(1))] = p
    return found


def _run_step_diagnostics(
    ckpt_paths: dict[str, Path],
    batch: dict[str, torch.Tensor],
    step: int,
    named_fields: dict[int, str],
    device: torch.device,
) -> dict[str, Any]:
    """Run diagnostics for the paired models at one training step."""
    step_result: dict[str, Any] = {}

    for model_name, ckpt_path in ckpt_paths.items():
        ckpt  = _load_checkpoint(ckpt_path)
        model = _rebuild_model(ckpt).to(device)

        field_ids = batch["field_ids"]
        attn_mask = batch["attention_mask"]
        valid     = _valid_mask(field_ids, attn_mask, named_fields)

        with torch.no_grad():
            output = model(
                batch["input_ids"],
                field_ids,
                attention_mask=attn_mask,
                need_weights=True,
            )

        attn_probs = output.attentions
        if not attn_probs:
            print(f"  WARNING: no attention weights for {model_name} step {step}")
            continue

        sfm = _same_field_mass(attn_probs, field_ids, valid)
        f2f = _field_to_field_matrix(attn_probs, field_ids, valid, named_fields)
        ent = _attention_entropy(attn_probs, valid)

        step_result[model_name] = {
            "step":                          step,
            "variant":                       model_name,
            "same_field_mass_per_layer":     sfm,
            "same_field_mass_avg":           sum(sfm) / len(sfm),
            "attention_entropy_per_layer":   ent,
            "attention_entropy_avg":         sum(ent) / len(ent),
            "field_to_field_per_layer":      f2f,
        }

    return step_result


def _print_step_summary(step: int, step_result: dict[str, Any]) -> None:
    print(f"\n  step {step:4d} │ {'model':<10} {'sfm_avg':>8} {'sfm_L0':>8} {'sfm_L3':>8} {'ent_avg':>8}")
    print(f"          │ {'-'*46}")
    for m in MODELS:
        if m not in step_result:
            continue
        r = step_result[m]
        sfm = r["same_field_mass_per_layer"]
        print(f"          │ {m:<10} {r['same_field_mass_avg']:>8.4f} {sfm[0]:>8.4f} {sfm[-1]:>8.4f} {r['attention_entropy_avg']:>8.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Time-series attention diagnostics.")
    parser.add_argument("--run-root",  type=Path, required=True,
                        help="Root directory containing {model}_seed{seed}/ subdirs.")
    parser.add_argument("--seed",      type=int,  required=True)
    parser.add_argument("--val-jsonl", type=Path, default=None)
    parser.add_argument("--n-examples", type=int, default=32)
    parser.add_argument("--synthetic",  action="store_true")
    parser.add_argument("--field-vocab-json", type=Path, default=None,
                        help="Optional field_vocab.json produced during data preparation.")
    parser.add_argument("--out-dir",   type=Path,
                        default=Path("runs/diagnostics/timeseries"))
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args(argv)

    # ── Discover checkpoints ──────────────────────────────────────────────────
    print(f"Scanning {args.run_root} for seed {args.seed} diagnostic checkpoints...")
    all_ckpts = _find_diagnostic_checkpoints(args.run_root, args.seed)

    missing_models = [model for model in MODELS if not all_ckpts[model]]
    if missing_models:
        print(
            "ERROR: Missing diagnostic checkpoints for "
            + ", ".join(missing_models)
            + ". Both Baseline and SAAB are required.",
            file=sys.stderr,
        )
        return 1
    steps_per_model = [set(all_ckpts[model]) for model in MODELS]
    common_steps = sorted(set.intersection(*steps_per_model))
    if not common_steps:
        print("ERROR: Baseline and SAAB have no common checkpoint steps.", file=sys.stderr)
        return 1

    print(f"Found {len(common_steps)} common steps: {common_steps}")
    for m in MODELS:
        print(f"  {m}: {sorted(all_ckpts[m].keys())}")

    # ── Load fixed batch ──────────────────────────────────────────────────────
    if args.synthetic or args.val_jsonl is None:
        print(f"\nUsing synthetic batch ({args.n_examples} examples).")
        batch = _make_synthetic_batch(n=args.n_examples)
    else:
        if not args.val_jsonl.exists():
            print(f"ERROR: val_jsonl not found: {args.val_jsonl}", file=sys.stderr)
            return 1
        print(f"\nLoading {args.n_examples} examples from {args.val_jsonl}")
        batch = _load_real_examples(args.val_jsonl, args.n_examples)

    from structformer.utils.device import DeviceError, select_device

    try:
        selected = select_device(args.device, allow_cpu=args.allow_cpu)
    except DeviceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    device = torch.device(selected.name)
    batch = {name: tensor.to(device) for name, tensor in batch.items()}

    active_models = MODELS
    first_ckpt = _load_checkpoint(all_ckpts[active_models[0]][common_steps[0]])
    mask_field_id = int(first_ckpt["config"]["data"]["mask_field_id"])
    named_fields = _detect_named_fields(batch, mask_field_id, args.field_vocab_json)
    if not named_fields:
        print("ERROR: No named fields were found in the diagnostic batch.", file=sys.stderr)
        return 1
    print(f"Named fields detected: {named_fields}")

    # ── Run diagnostics at each step ─────────────────────────────────────────
    print(f"\nRunning diagnostics at {len(common_steps)} steps × {len(active_models)} models ({', '.join(active_models)})...")
    timeseries: dict[str, Any] = {
        "seed":       args.seed,
        "steps":      common_steps,
        "n_examples": args.n_examples,
        "synthetic":  args.synthetic or args.val_jsonl is None,
        "named_fields": {str(key): value for key, value in named_fields.items()},
        "by_step":    {},
    }

    for step in common_steps:
        ckpt_paths = {m: all_ckpts[m][step] for m in active_models}
        step_result = _run_step_diagnostics(
            ckpt_paths,
            batch,
            step,
            named_fields,
            device,
        )
        timeseries["by_step"][str(step)] = step_result
        _print_step_summary(step, step_result)

    # ── Save output ───────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "timeseries_diagnostics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(timeseries, f, indent=2)
        f.write("\n")
    print(f"\nSaved: {out_path}")

    # ── Print final cross-step summary ────────────────────────────────────────
    print("\n══ Same-field mass L3 (last layer) across training ══════════")
    print(f"  {'step':>6} │ {'baseline':>10} {'saab':>10}")
    print(f"  {'-'*44}")
    for step in common_steps:
        row = timeseries["by_step"][str(step)]
        vals = []
        for m in MODELS:
            sfm = row.get(m, {}).get("same_field_mass_per_layer", [])
            vals.append(f"{sfm[-1]:>10.4f}" if sfm else f"{'n/a':>10}")
        print(f"  {step:>6} │ {''.join(vals)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
