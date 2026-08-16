"""Legacy per-layer logit-sensitivity diagnostic across checkpoints.

Loads step_XXXX.pt diagnostic checkpoints for baseline and SAAB, runs a
forward+backward pass on a fixed validation batch, and records the L2 norm
of gradients at each layer's attention Q/K/V projections.

This script uses logits.sum(), not the MSM training loss, and it does not mask
field IDs. Its output is therefore a gradient-based sensitivity of aggregate
logits and must not be interpreted as an optimization-gradient measurement.
For the corrected experiment, enable --log-layer-gradients during MSM training
and analyze the resulting logs with scripts/analyze_training_gradients.py.

Metric per layer
────────────────
  grad_norm[L] = mean(||dL/dW_q||, ||dL/dW_k||, ||dL/dW_v||)

where W_q, W_k, W_v are the Q/K/V projection weight matrices in layer L.

Output: gradient_diagnostics.json

Usage:
    PYTHONPATH=src python scripts/diag_gradients.py \\
        --run-root /content/runs/dbpedia_phase2_diag \\
        --seed 1001 \\
        --val-jsonl data/processed/benchmark/dbpedia_msm/val.jsonl \\
        --n-examples 32 \\
        --out-dir /content/outputs/diagnostics/gradients

    # Synthetic fallback (no val.jsonl needed):
    PYTHONPATH=src python scripts/diag_gradients.py \\
        --run-root /content/runs/dbpedia_phase2_diag \\
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

sys.path.insert(0, str(Path(__file__).parent))
from diag_attention import (
    _load_checkpoint,
    _load_real_examples,
    _make_synthetic_batch,
    _rebuild_model,
)

MODELS = ["baseline", "saab"]


# ── Checkpoint discovery ──────────────────────────────────────────────────────

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


# ── Gradient measurement ──────────────────────────────────────────────────────

def _measure_gradient_norms(
    model: Any,
    batch: dict[str, torch.Tensor],
) -> list[float]:
    """Return legacy aggregate-logit sensitivity per attention layer."""
    model.train()
    model.zero_grad()

    output = model(
        batch["input_ids"],
        batch["field_ids"],
        attention_mask=batch["attention_mask"],
        need_weights=False,
    )

    # Surrogate loss: sum of all logits — differentiable through the full graph
    loss = output.logits.sum()
    loss.backward()

    norms: list[float] = []
    for layer in model.encoder.layers:
        attn = layer.attention
        layer_norms = []
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
            if proj.weight.grad is not None:
                layer_norms.append(proj.weight.grad.norm().item())
        norms.append(sum(layer_norms) / len(layer_norms) if layer_norms else 0.0)

    model.zero_grad()
    return norms


# ── Per-step driver ───────────────────────────────────────────────────────────

def _run_step(
    ckpt_paths: dict[str, Path],
    batch: dict[str, torch.Tensor],
    step: int,
) -> dict[str, Any]:
    step_result: dict[str, Any] = {}
    for model_name, ckpt_path in ckpt_paths.items():
        ckpt  = _load_checkpoint(ckpt_path)
        model = _rebuild_model(ckpt)
        norms = _measure_gradient_norms(model, batch)
        step_result[model_name] = {
            "step":                   step,
            "variant":                model_name,
            "grad_norm_per_layer":    norms,
            "grad_norm_last_layer":   norms[-1] if norms else None,
            "grad_norm_ratio_L3_L2":  (norms[-1] / norms[-2]) if len(norms) >= 2 and norms[-2] > 0 else None,
        }
    return step_result


def _print_step_summary(step: int, step_result: dict[str, Any]) -> None:
    num_layers = max(len(r.get("grad_norm_per_layer", [])) for r in step_result.values())
    header_layers = "  ".join(f"{'L'+str(i):>8}" for i in range(num_layers))
    print(f"\n  step {step:4d} │ {'model':<10} {header_layers}  {'L3/L2':>8}")
    print(f"          │ {'-' * (12 + 10 * num_layers)}")
    for m in MODELS:
        if m not in step_result:
            continue
        r     = step_result[m]
        norms = r["grad_norm_per_layer"]
        ratio = r.get("grad_norm_ratio_L3_L2")
        norm_str  = "  ".join(f"{n:>8.4f}" for n in norms)
        ratio_str = f"{ratio:>8.3f}" if ratio is not None else f"{'n/a':>8}"
        print(f"          │ {m:<10} {norm_str}  {ratio_str}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Legacy aggregate-logit sensitivity diagnostic.")
    parser.add_argument("--run-root",    type=Path, required=True,
                        help="Root containing {model}_seed{seed}/ subdirs with step_*.pt checkpoints.")
    parser.add_argument("--seed",        type=int,  required=True)
    parser.add_argument("--val-jsonl",   type=Path, default=None)
    parser.add_argument("--n-examples",  type=int,  default=32)
    parser.add_argument("--synthetic",   action="store_true")
    parser.add_argument("--out-dir",     type=Path,
                        default=Path("runs/diagnostics/gradients"))
    args = parser.parse_args(argv)
    print(
        "WARNING: this legacy script measures gradients of logits.sum() on "
        "unmasked inputs. It does not measure MSM training-loss gradients."
    )

    # ── Discover checkpoints ──────────────────────────────────────────────────
    print(f"Scanning {args.run_root} for seed {args.seed} diagnostic checkpoints...")
    all_ckpts = _find_diagnostic_checkpoints(args.run_root, args.seed)

    steps_per_model = [set(v.keys()) for v in all_ckpts.values() if v]
    if not steps_per_model:
        print("ERROR: No diagnostic checkpoints found.", file=sys.stderr)
        return 1
    common_steps = sorted(set.intersection(*steps_per_model))
    if not common_steps:
        print("ERROR: No steps with checkpoints for both models.", file=sys.stderr)
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

    # ── Run gradient diagnostics at each step ─────────────────────────────────
    print(f"\nMeasuring gradient norms at {len(common_steps)} steps × {len(MODELS)} models...")
    results: dict[str, Any] = {
        "seed":       args.seed,
        "steps":      common_steps,
        "n_examples": args.n_examples,
        "synthetic":  args.synthetic or args.val_jsonl is None,
        "metric":     "legacy aggregate-logit sensitivity: mean Q/K/V norm, objective=logits.sum()",
        "by_step":    {},
    }

    for step in common_steps:
        ckpt_paths = {m: all_ckpts[m][step] for m in MODELS if step in all_ckpts[m]}
        step_result = _run_step(ckpt_paths, batch, step)
        results["by_step"][str(step)] = step_result
        _print_step_summary(step, step_result)

    # ── Save ──────────────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "gradient_diagnostics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ── Final summary: L3/L2 ratio across steps ───────────────────────────────
    print("\n══ L3/L2 gradient ratio across training ═══════════════════════")
    print(f"  {'step':>6} │ {'baseline':>10} {'saab':>10}")
    print(f"  {'-'*30}")
    for step in common_steps:
        row = results["by_step"][str(step)]
        vals = []
        for m in MODELS:
            r = row.get(m, {})
            ratio = r.get("grad_norm_ratio_L3_L2")
            vals.append(f"{ratio:>10.3f}" if ratio is not None else f"{'n/a':>10}")
        print(f"  {step:>6} │ {''.join(vals)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
