"""Attention diagnostics for baseline and SAAB checkpoints.

Loads paired Baseline and SAAB checkpoints, runs a fixed batch of examples
through each with need_weights=True, and computes compact
structural-attention metrics:

  - Same-field attention mass per layer
  - Field-to-field attention mass matrix (title/content)
  - Attention entropy per layer
  - SAAB structural bias statistics

Outputs a JSON file and prints a human-readable summary.

Usage (real data):
    PYTHONPATH=src python scripts/diag_attention.py \\
        --baseline-ckpt runs/dbpedia_multiseed/baseline_seed1001/checkpoints/latest.pt \\
        --saab-ckpt     runs/dbpedia_multiseed/saab_seed1001/checkpoints/latest.pt \\
        --val-jsonl     /path/to/processed/dbpedia/val.jsonl \\
        --n-examples    32 \\
        --device         cuda \\
        --out-dir        outputs/diagnostics/attention

Usage (synthetic fallback, no val.jsonl needed):
    PYTHONPATH=src python scripts/diag_attention.py \\
        --baseline-ckpt runs/example/baseline/checkpoints/latest.pt \\
        --saab-ckpt     runs/example/saab/checkpoints/latest.pt \\
        --synthetic --device cpu --allow-cpu
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# ── Field ID constants ────────────────────────────────────────────────────────
PAD_FIELD  = 0
NONE_FIELD = 1
UNK_FIELD  = 2
SYNTHETIC_FIELD_A = 3
SYNTHETIC_FIELD_B = 4
# Named fields are detected dynamically from the data + checkpoint config,
# so this works for any number of fields (DBpedia: 2, PubMed: 3, etc.)


# ── Model reconstruction ──────────────────────────────────────────────────────

def _rebuild_model(ckpt: dict[str, Any]) -> Any:
    """Reconstruct StructuredTransformerModel from a saved checkpoint."""
    from structformer.models import StructuredTransformerModel, TransformerConfig

    cfg  = ckpt["config"]
    data = cfg["data"]
    mc   = cfg["model_config"]
    saved_token_embeddings = ckpt["model_state_dict"][
        "embeddings.token_embeddings.weight"
    ]

    transformer_config = TransformerConfig(
        vocab_size        = int(
            data.get("vocab_size") or saved_token_embeddings.shape[0]
        ),
        field_vocab_size  = int(data["field_vocab_size"]),
        max_length        = int(data["max_length"]),
        variant           = str(cfg["model"]),
        num_labels        = int(data["num_labels"]),
        d_model           = int(mc["d_model"]),
        num_layers        = int(mc["num_layers"]),
        num_heads         = int(mc["num_heads"]),
        ff_dim            = int(mc["ff_dim"]),
        dropout           = float(mc["dropout"]),
        pad_token_id      = int(data["pad_token_id"]),
        scale_embeddings  = bool(mc.get("scale_embeddings", False)),
        saab_field_weight = float(mc.get("saab_field_weight", 1.0)),
        saab_layer_mask   = tuple(float(x) for x in mc.get("saab_layer_mask", [])),
        saab_shuffle_bias = bool(mc.get("saab_shuffle_bias", False)),
        saab_shuffle_seed = int(mc.get("saab_shuffle_seed", 0)),
    )
    model = StructuredTransformerModel(transformer_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_real_examples(val_jsonl: Path, n: int, max_length: int = 256) -> dict[str, torch.Tensor]:
    """Load first n examples from val.jsonl and collate into tensors."""
    records = []
    with open(val_jsonl) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records found in {val_jsonl}")

    def _pad(seqs: list[list[int]], pad_val: int) -> torch.Tensor:
        length = max(len(s) for s in seqs)
        length = min(length, max_length)
        out = torch.full((len(seqs), length), pad_val, dtype=torch.long)
        for i, s in enumerate(seqs):
            s = s[:length]
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    input_ids    = _pad([r["input_ids"]  for r in records], 0)
    field_ids    = _pad([r["field_ids"]  for r in records], 0)
    attn_mask    = _pad([r["attention_mask"] for r in records], 0).bool()
    return {"input_ids": input_ids, "field_ids": field_ids, "attention_mask": attn_mask}


def _make_synthetic_batch(n: int = 32, seq_len: int = 128, vocab_size: int = 30000) -> dict[str, torch.Tensor]:
    """Simple synthetic batch: first half title, second half content."""
    split = seq_len // 2
    field_ids = torch.cat([
        torch.full((n, split), SYNTHETIC_FIELD_A, dtype=torch.long),
        torch.full((n, seq_len - split), SYNTHETIC_FIELD_B, dtype=torch.long),
    ], dim=1)
    midpoint   = vocab_size // 2
    input_ids  = torch.cat([
        torch.randint(2, midpoint,  (n, split)),
        torch.randint(midpoint, vocab_size, (n, seq_len - split)),
    ], dim=1)
    attn_mask  = torch.ones(n, seq_len, dtype=torch.long)
    return {"input_ids": input_ids, "field_ids": field_ids, "attention_mask": attn_mask}


# ── Metric helpers ────────────────────────────────────────────────────────────

def _detect_named_fields(
    batch: dict[str, torch.Tensor],
    mask_field_id: int,
    field_vocab_json: Path | None = None,
) -> dict[int, str]:
    """Build {field_id: name} from data, excluding padding/special/mask tokens.

    If field_vocab_json is provided, uses its names. Otherwise uses generic
    names like 'field_3'. Works for any number of fields.
    """
    all_ids = batch["field_ids"].unique().tolist()
    named_ids = sorted(int(fid) for fid in all_ids
                       if int(fid) > UNK_FIELD and int(fid) != mask_field_id)

    if field_vocab_json and field_vocab_json.exists():
        with open(field_vocab_json) as f:
            vocab_data = json.load(f)
        # field_vocab maps name→id; invert to id→name
        id_to_name = {v: k for k, v in vocab_data.get("field_vocab", {}).items()}
        return {fid: id_to_name.get(fid, f"field_{fid}") for fid in named_ids}

    return {fid: f"field_{fid}" for fid in named_ids}


def _valid_mask(
    field_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    named_fields: dict[int, str],
) -> torch.Tensor:
    """Boolean mask: positions that are non-padding and in a named field."""
    named = torch.zeros_like(field_ids, dtype=torch.bool)
    for fid in named_fields:
        named |= field_ids.eq(fid)
    return named & attention_mask.bool()


def _same_field_mass(
    attn_probs: list[torch.Tensor],
    field_ids:  torch.Tensor,
    valid:      torch.Tensor,
) -> list[float]:
    """Average fraction of attention weight staying within the same field, per layer.

    attn_probs: list of [batch, heads, seq, seq] tensors (one per layer).
    Returns one float per layer.
    """
    results = []
    for layer_attn in attn_probs:
        # layer_attn: [B, H, S, S]
        B, H, S, _ = layer_attn.shape
        # same_field_mask[b, i, j] = 1 iff field_ids[b,i] == field_ids[b,j]
        same = field_ids.unsqueeze(2).eq(field_ids.unsqueeze(1)).float()  # [B, S, S]
        same = same.unsqueeze(1)  # [B, 1, S, S]

        # only sum over valid key positions
        key_valid = valid.float().unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
        attn_valid = layer_attn * key_valid  # zero out padding keys
        same_mass  = (attn_valid * same).sum(dim=-1)  # [B, H, S]

        # average over valid query positions only
        q_valid = valid.float().unsqueeze(1)  # [B, 1, S]
        total_q = q_valid.sum(dim=-1).clamp_min(1.0)  # [B, 1]
        layer_mean = (same_mass * q_valid).sum(dim=-1) / total_q  # [B, H]
        results.append(layer_mean.mean().item())

    return results


def _field_to_field_matrix(
    attn_probs:    list[torch.Tensor],
    field_ids:     torch.Tensor,
    valid:         torch.Tensor,
    named_fields:  dict[int, str],
) -> dict[str, list[list[float]]]:
    """Average attention mass from each named field to each named field.

    Returns a dict mapping layer index (str) to an NxN list (N = number of
    named fields). Works for any number of fields.
    """
    field_list  = list(named_fields.keys())
    field_names = list(named_fields.values())
    n_fields    = len(field_list)

    layer_matrices = {}
    for layer_idx, layer_attn in enumerate(attn_probs):
        B, H, S, _ = layer_attn.shape
        matrix = [[0.0] * n_fields for _ in range(n_fields)]
        counts = [[0]   * n_fields for _ in range(n_fields)]

        key_valid = valid.float()  # [B, S]

        for si, src_fid in enumerate(field_list):
            src_mask = field_ids.eq(src_fid) & valid  # [B, S]  query positions
            for di, dst_fid in enumerate(field_list):
                dst_mask = field_ids.eq(dst_fid) & valid  # [B, S]  key positions
                # attn from src queries to dst keys
                # layer_attn[:, :, src_positions, dst_positions]
                dst_m = dst_mask.float().unsqueeze(1).unsqueeze(2)   # [B, 1, 1, S]
                mass  = (layer_attn * dst_m).sum(dim=-1)             # [B, H, S]
                # average over src query positions
                src_m = src_mask.float().unsqueeze(1)                # [B, 1, S]
                n_src = src_m.sum(dim=-1).clamp_min(1.0)             # [B, 1]
                val   = (mass * src_m).sum(dim=-1) / n_src           # [B, H]
                if src_mask.any():
                    matrix[si][di] = val.mean().item()
                    counts[si][di] = 1

        layer_matrices[str(layer_idx)] = {
            "fields": field_names,
            "matrix": matrix,
        }

    return layer_matrices


def _attention_entropy(
    attn_probs: list[torch.Tensor],
    valid:      torch.Tensor,
) -> list[float]:
    """Mean attention entropy per layer (nats), over valid query positions."""
    results = []
    for layer_attn in attn_probs:
        B, H, S, _ = layer_attn.shape
        eps = 1e-10
        entropy = -(layer_attn * (layer_attn + eps).log()).sum(dim=-1)  # [B, H, S]
        q_valid = valid.float().unsqueeze(1)                            # [B, 1, S]
        total_q = q_valid.sum(dim=-1).clamp_min(1.0)                   # [B, 1]
        mean_h  = (entropy * q_valid).sum(dim=-1) / total_q            # [B, H]
        results.append(mean_h.mean().item())
    return results




def _saab_bias_stats(
    model: Any,
    batch: dict[str, torch.Tensor],
    named_fields: dict[int, str],
) -> dict[str, float] | None:
    """Mean SAAB bias for same-field and cross-field token pairs."""
    if model.config.variant != "saab":
        return None
    from structformer.models.bias import SAABWeights, build_saab_bias

    field_ids = batch["field_ids"]
    valid     = _valid_mask(field_ids, batch["attention_mask"], named_fields)
    bias      = build_saab_bias(field_ids)   # [B, S, S]

    same = field_ids.unsqueeze(2).eq(field_ids.unsqueeze(1))   # [B, S, S]
    pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)        # [B, S, S]
    same_valid  = (same  & pair_valid).float()
    cross_valid = (~same & pair_valid).float()

    def _mean(mask: torch.Tensor) -> float:
        n = mask.sum()
        return (bias * mask).sum().item() / n.clamp_min(1).item()

    return {
        "same_field_mean":  _mean(same_valid),
        "cross_field_mean": _mean(cross_valid),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_diagnostics(
    ckpt_paths:       dict[str, Path],
    batch:            dict[str, torch.Tensor],
    out_dir:          Path,
    field_vocab_json: Path | None = None,
    device: torch.device | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cpu")
    batch = {name: tensor.to(device) for name, tensor in batch.items()}

    all_results: dict[str, Any] = {}

    # Detect named fields once from the first checkpoint + data
    _first_ckpt = _load_checkpoint(next(iter(ckpt_paths.values())))
    _mask_fid   = int(_first_ckpt["config"]["data"]["mask_field_id"])
    named_fields = _detect_named_fields(batch, _mask_fid, field_vocab_json)
    print(f"Named fields detected: { {k: v for k, v in named_fields.items()} }")

    for variant, ckpt_path in ckpt_paths.items():
        print(f"\n── {variant} ({ckpt_path.name}) ──────────────────────────")
        ckpt  = _load_checkpoint(ckpt_path)
        model = _rebuild_model(ckpt).to(device)

        field_ids    = batch["field_ids"]
        attn_mask    = batch["attention_mask"]
        valid        = _valid_mask(field_ids, attn_mask, named_fields)

        with torch.no_grad():
            output = model(
                batch["input_ids"],
                field_ids,
                attention_mask=attn_mask,
                need_weights=True,
            )

        attn_probs = output.attentions   # list of [B, H, S, S] per layer
        if not attn_probs:
            print("  WARNING: no attention weights returned")
            continue

        n_layers = len(attn_probs)

        # --- metrics ----------------------------------------------------------
        sfm    = _same_field_mass(attn_probs, field_ids, valid)
        f2f    = _field_to_field_matrix(attn_probs, field_ids, valid, named_fields)
        ent    = _attention_entropy(attn_probs, valid)
        sbias  = _saab_bias_stats(model, batch, named_fields) if variant == "saab" else {}

        result: dict[str, Any] = {
            "variant":            variant,
            "seed":               ckpt["config"]["seed"],
            "step":               ckpt["step"],
            "n_examples":         batch["input_ids"].shape[0],
            "n_layers":           n_layers,
            "same_field_mass_per_layer":    sfm,
            "same_field_mass_avg":          sum(sfm) / len(sfm),
            "attention_entropy_per_layer":  ent,
            "attention_entropy_avg":        sum(ent) / len(ent),
            "field_to_field_per_layer":     f2f,
            "saab_bias_stats":              sbias,
        }

        # --- print summary ----------------------------------------------------
        print(f"  same-field attention mass (per layer): " +
              "  ".join(f"L{i}={v:.3f}" for i, v in enumerate(sfm)))
        print(f"  avg same-field mass:  {result['same_field_mass_avg']:.4f}")
        print(f"  attn entropy (per layer): " +
              "  ".join(f"L{i}={v:.3f}" for i, v in enumerate(ent)))
        print(f"  avg entropy:          {result['attention_entropy_avg']:.4f}")


        if sbias:
            print(f"  SAAB bias  same-field: {sbias['same_field_mean']:.4f}  "
                  f"cross-field: {sbias['cross_field_mean']:.4f}")

        # field-to-field: last layer summary
        last = f2f[str(n_layers - 1)]
        fields = last["fields"]
        mat    = last["matrix"]
        print(f"  field-to-field (layer {n_layers-1}):")
        header = "         " + "  ".join(f"→{f[:4]:4s}" for f in fields)
        print(f"  {header}")
        for i, src in enumerate(fields):
            row = "  ".join(f"{mat[i][j]:.4f}" for j in range(len(fields)))
            print(f"  {src[:4]:4s}  |  {row}")

        all_results[variant] = result

    # ── cross-model comparison summary ────────────────────────────────────────
    print("\n══ Cross-model summary ══════════════════════════════════════")
    print(f"  {'metric':<35} {'baseline':>10} {'saab':>10}")
    print("  " + "-" * 68)

    metrics_to_compare = [
        ("avg same-field mass",    "same_field_mass_avg"),
        ("avg attention entropy",  "attention_entropy_avg"),
    ]
    for label, key in metrics_to_compare:
        row = ""
        for v in ["baseline", "saab"]:
            val = all_results.get(v, {}).get(key, float("nan"))
            row += f"  {val:>10.4f}"
        print(f"  {label:<35}{row}")

    # ── write output ──────────────────────────────────────────────────────────
    out_path = out_dir / "attention_diagnostics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
        f.write("\n")
    print(f"\n  Saved: {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Attention diagnostics for baseline and SAAB.")
    parser.add_argument("--baseline-ckpt", type=Path, required=True)
    parser.add_argument("--saab-ckpt",     type=Path, required=True)
    parser.add_argument("--val-jsonl",  type=Path, default=None,
                        help="Prepared val JSONL. If omitted, uses synthetic batch.")
    parser.add_argument("--n-examples", type=int, default=32,
                        help="Number of examples to run diagnostics on.")
    parser.add_argument("--synthetic",  action="store_true",
                        help="Force synthetic batch even if --val-jsonl is provided.")
    parser.add_argument("--field-vocab-json", type=Path, default=None,
                        help="field_vocab.json from data prep (provides field names). "
                             "If omitted, field names are auto-detected as 'field_<id>'.")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("runs/diagnostics/attention"))
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args(argv)

    ckpt_paths = {
        "baseline": args.baseline_ckpt,
        "saab":     args.saab_ckpt,
    }
    for name, p in ckpt_paths.items():
        if not p.exists():
            print(f"ERROR: {name} checkpoint not found: {p}", file=sys.stderr)
            return 1

    if args.synthetic or args.val_jsonl is None:
        print("Using synthetic batch (32 examples, 128 tokens each).")
        batch = _make_synthetic_batch(n=args.n_examples)
    else:
        if not args.val_jsonl.exists():
            print(f"ERROR: val_jsonl not found: {args.val_jsonl}", file=sys.stderr)
            return 1
        print(f"Loading {args.n_examples} examples from {args.val_jsonl}")
        batch = _load_real_examples(args.val_jsonl, args.n_examples)

    from structformer.utils.device import DeviceError, select_device

    try:
        selected = select_device(args.device, allow_cpu=args.allow_cpu)
    except DeviceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    run_diagnostics(
        ckpt_paths,
        batch,
        args.out_dir,
        args.field_vocab_json,
        torch.device(selected.name),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
