#!/usr/bin/env python3
"""Print copy-pasteable commands for the eight-seed DBpedia experiment."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path


DEFAULT_SEEDS = "0,7,42,99,123,256,1001,2024"
DEFAULT_MODELS = "baseline,saab"
DEFAULT_STEPS = "1,50,100,200,300,500"
VALID_MODELS = {"baseline", "saab"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print commands for the paper's eight-seed DBpedia experiment."
    )
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("runs/dbpedia_multiseed"))
    parser.add_argument("--table-out-dir", type=Path, default=Path("outputs/tables/dbpedia_multiseed"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("dbpedia/configs/msm_dbpedia_full_recipe.yaml"),
    )
    parser.add_argument("--seeds", default=DEFAULT_SEEDS, help=f"Comma-separated seeds. Default: {DEFAULT_SEEDS}")
    parser.add_argument("--models", default=DEFAULT_MODELS, help=f"Comma-separated models. Default: {DEFAULT_MODELS}")
    parser.add_argument("--steps", default=DEFAULT_STEPS, help=f"Comma-separated table steps. Default: {DEFAULT_STEPS}")
    parser.add_argument("--python", default="python", help="Python executable to use in printed commands.")
    parser.add_argument("--env-prefix", default="PYTHONPATH=src", help="Shell prefix for printed commands.")
    parser.add_argument("--no-table", action="store_true", help="Do not print the table-generation command.")
    args = parser.parse_args(argv)

    seeds = _parse_int_list(args.seeds, name="seeds")
    models = _parse_models(args.models)

    commands = [
        train_command(
            python=args.python,
            env_prefix=args.env_prefix,
            config=args.config,
            model=model,
            seed=seed,
            train_jsonl=args.train_jsonl,
            val_jsonl=args.val_jsonl,
            tokenizer_json=args.tokenizer_json,
            run_root=args.run_root,
        )
        for seed in seeds
        for model in models
    ]

    print(f"# DBpedia eight-seed train commands: {len(commands)} runs")
    print(f"# seeds: {', '.join(str(seed) for seed in seeds)}")
    print(f"# models: {', '.join(models)}")
    print()
    for command in commands:
        print(command)
        print()

    if not args.no_table:
        print("# Table command, after all runs finish")
        print(
            table_command(
                python=args.python,
                env_prefix=args.env_prefix,
                run_root=args.run_root,
                table_out_dir=args.table_out_dir,
                steps=args.steps,
            )
        )
    return 0


def train_command(
    *,
    python: str,
    env_prefix: str,
    config: Path,
    model: str,
    seed: int,
    train_jsonl: Path,
    val_jsonl: Path,
    tokenizer_json: Path,
    run_root: Path,
) -> str:
    run_dir = run_root / f"{model}_seed{seed}"
    parts = [
        python,
        "-m",
        "structformer.training.train_msm",
        "--config",
        str(config),
        "--model",
        model,
        "--seed",
        str(seed),
        "--train-jsonl",
        str(train_jsonl),
        "--val-jsonl",
        str(val_jsonl),
        "--tokenizer-json",
        str(tokenizer_json),
        "--run-dir",
        str(run_dir),
    ]
    return _with_env_prefix(env_prefix, parts)


def table_command(*, python: str, env_prefix: str, run_root: Path, table_out_dir: Path, steps: str) -> str:
    parts = [
        python,
        "scripts/make_tables.py",
        "--runs",
        str(run_root),
        "--out-dir",
        str(table_out_dir),
        "--steps",
        steps,
    ]
    return _with_env_prefix(env_prefix, parts)


def _parse_int_list(value: str, *, name: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise SystemExit(f"{name} must contain at least one value")
    try:
        return [int(item) for item in items]
    except ValueError as exc:
        raise SystemExit(f"{name} must be a comma-separated list of integers") from exc


def _parse_models(value: str) -> list[str]:
    models = [item.strip() for item in value.split(",") if item.strip()]
    if not models:
        raise SystemExit("models must contain at least one value")
    invalid = [model for model in models if model not in VALID_MODELS]
    if invalid:
        raise SystemExit(f"models contains invalid values: {', '.join(invalid)}")
    return models


def _with_env_prefix(env_prefix: str, parts: list[str]) -> str:
    command = " ".join(shlex.quote(part) for part in parts)
    return f"{env_prefix} {command}" if env_prefix else command


if __name__ == "__main__":
    raise SystemExit(main())
