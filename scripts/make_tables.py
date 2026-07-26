#!/usr/bin/env python3
"""Generate CSV/Markdown tables from run metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from structformer.analysis.metrics import (  # noqa: E402
    discover_run_dirs,
    final_rows,
    parse_steps,
    read_all_metrics,
    rows_at_steps,
    write_csv,
    write_markdown,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate result tables from metrics.jsonl files.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories or parent directories to scan.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", default=None, help="Optional comma-separated steps, e.g. 10,50,100,500.")
    args = parser.parse_args(argv)

    run_dirs = discover_run_dirs(args.runs)
    if not run_dirs:
        print("No run directories with metrics.jsonl found.", file=sys.stderr)
        return 1

    rows = read_all_metrics(run_dirs)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    final = final_rows(rows)
    write_csv(final, args.out_dir / "final_metrics.csv")
    write_markdown(final, args.out_dir / "final_metrics.md")

    steps = parse_steps(args.steps)
    if steps:
        selected = rows_at_steps(rows, steps)
        write_csv(selected, args.out_dir / "step_metrics.csv")
        write_markdown(selected, args.out_dir / "step_metrics.md")

    print(f"Wrote tables for {len(run_dirs)} runs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

