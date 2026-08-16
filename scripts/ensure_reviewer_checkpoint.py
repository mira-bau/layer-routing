#!/usr/bin/env python3
"""Verify or reproduce one reviewer-analysis checkpoint.

This command is intentionally suitable for a Colab launcher: it inspects a
checkpoint without eagerly materializing its tensor storage, keeps active I/O
on local disk, prints every state transition, and copies only a completed
checkpoint back to persistent storage.
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class ExpectedCheckpoint:
    variant: str
    seed: int
    step: int
    max_length: int
    d_model: int = 768
    num_layers: int = 4
    num_heads: int = 6
    ff_dim: int = 3072


def load_checkpoint_identity(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None

    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, TypeError):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    identity = {
        "variant": str(config["model"]),
        "seed": int(config["seed"]),
        "step": int(checkpoint["step"]),
        "max_steps": int(config["training"]["max_steps"]),
        "max_length": int(config["data"]["max_length"]),
        "d_model": int(config["model_config"]["d_model"]),
        "num_layers": int(config["model_config"]["num_layers"]),
        "num_heads": int(config["model_config"]["num_heads"]),
        "ff_dim": int(config["model_config"]["ff_dim"]),
    }
    del checkpoint
    gc.collect()
    return identity


def validate_identity(path: Path, identity: dict[str, Any], expected: ExpectedCheckpoint) -> bool:
    expected_static = {
        "variant": expected.variant,
        "seed": expected.seed,
        "max_steps": expected.step,
        "max_length": expected.max_length,
        "d_model": expected.d_model,
        "num_layers": expected.num_layers,
        "num_heads": expected.num_heads,
        "ff_dim": expected.ff_dim,
    }
    actual_static = {key: value for key, value in identity.items() if key != "step"}
    if actual_static != expected_static:
        raise RuntimeError(f"Checkpoint mismatch at {path}: {identity}")
    if identity["step"] > expected.step:
        raise RuntimeError(f"Checkpoint is beyond expected step {expected.step} at {path}: {identity}")
    return identity["step"] == expected.step


def _unused_preserved_path(run_dir: Path, label: str) -> Path:
    candidate = run_dir.with_name(f"{run_dir.name}_{label}")
    suffix = 2
    while candidate.exists():
        candidate = run_dir.with_name(f"{run_dir.name}_{label}_{suffix}")
        suffix += 1
    return candidate


def _preserve_run_dir(run_dir: Path, label: str) -> Path | None:
    if not run_dir.exists():
        return None
    destination = _unused_preserved_path(run_dir, label)
    print(f"PRESERVING existing local run: {run_dir} -> {destination}", flush=True)
    shutil.move(str(run_dir), str(destination))
    return destination


def ensure_checkpoint(
    *,
    dataset: str,
    expected: ExpectedCheckpoint,
    local_checkpoint: Path,
    persistent_checkpoint: Path,
    config: Path,
    run_dir: Path,
    train_jsonl: Path,
    val_jsonl: Path,
    tokenizer_json: Path,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> str:
    print(f"CHECKING {dataset} {expected.variant}", flush=True)

    if not local_checkpoint.is_file() and run_dir.exists() and any(run_dir.iterdir()):
        _preserve_run_dir(run_dir, "interrupted_without_checkpoint")

    if not local_checkpoint.is_file() and persistent_checkpoint.is_file():
        local_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        print(f"COPYING saved checkpoint to local storage: {persistent_checkpoint}", flush=True)
        shutil.copy2(persistent_checkpoint, local_checkpoint)

    identity = load_checkpoint_identity(local_checkpoint)
    if identity is not None and validate_identity(local_checkpoint, identity, expected):
        print(f"VERIFIED complete {dataset} {expected.variant}: step {expected.step}/{expected.step}", flush=True)
        return "reused"

    if identity is not None:
        print(
            f"INCOMPLETE {dataset} {expected.variant}: step {identity['step']}/{expected.step}; "
            "an exact continuation is unavailable",
            flush=True,
        )
        _preserve_run_dir(run_dir, f"incomplete_step{identity['step']}")

    run_dir.mkdir(parents=True, exist_ok=True)
    command: Sequence[str] = [
        sys.executable,
        "-u",
        "-m",
        "structformer.training.train_msm",
        "--config",
        str(config),
        "--model",
        expected.variant,
        "--seed",
        str(expected.seed),
        "--train-jsonl",
        str(train_jsonl),
        "--val-jsonl",
        str(val_jsonl),
        "--tokenizer-json",
        str(tokenizer_json),
        "--run-dir",
        str(run_dir),
        "--max-steps",
        str(expected.step),
    ]
    print(
        f"STARTING {dataset} {expected.variant}: step 1/{expected.step} on local storage",
        flush=True,
    )
    result = runner(command, env={**os.environ, "PYTHONPATH": "src"})
    if result.returncode != 0:
        raise RuntimeError(f"Training failed: {dataset} {expected.variant} (exit {result.returncode})")

    final_identity = load_checkpoint_identity(local_checkpoint)
    if final_identity is None or not validate_identity(local_checkpoint, final_identity, expected):
        raise RuntimeError(f"Training ended without the required checkpoint: {local_checkpoint}")

    persistent_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print(f"SAVING completed checkpoint to Drive: {persistent_checkpoint}", flush=True)
    shutil.copy2(local_checkpoint, persistent_checkpoint)
    print(f"VERIFIED complete {dataset} {expected.variant}: step {expected.step}/{expected.step}", flush=True)
    return "trained"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", required=True, choices=["baseline", "saab"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-steps", required=True, type=int)
    parser.add_argument("--max-length", required=True, type=int)
    parser.add_argument("--local-checkpoint", required=True, type=Path)
    parser.add_argument("--persistent-checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--val-jsonl", required=True, type=Path)
    parser.add_argument("--tokenizer-json", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_checkpoint(
        dataset=args.dataset,
        expected=ExpectedCheckpoint(
            variant=args.variant,
            seed=args.seed,
            step=args.max_steps,
            max_length=args.max_length,
        ),
        local_checkpoint=args.local_checkpoint,
        persistent_checkpoint=args.persistent_checkpoint,
        config=args.config,
        run_dir=args.run_dir,
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        tokenizer_json=args.tokenizer_json,
    )


if __name__ == "__main__":
    main()
