"""Utilities for reading run metrics and generating small tables."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


MetricRow = dict[str, Any]

TABLE_COLUMNS = [
    "run",
    "task",
    "model",
    "seed",
    "phase",
    "split",
    "step",
    "loss",
    "accuracy",
    "device",
    "elapsed_seconds",
]


def discover_run_dirs(paths: Iterable[str | Path]) -> list[Path]:
    """Return run directories containing `metrics.jsonl`.

    Each input path may be a run directory or a parent directory containing run
    directories. Recursive discovery keeps Colab/local run layouts flexible.
    """

    discovered: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if (path / "metrics.jsonl").exists():
            discovered.append(path)
            continue
        if path.exists():
            discovered.extend(sorted(candidate.parent for candidate in path.rglob("metrics.jsonl")))
    return sorted(dict.fromkeys(discovered))


def read_metrics(run_dir: str | Path) -> list[MetricRow]:
    run_path = Path(run_dir)
    metrics_path = run_path / "metrics.jsonl"
    rows: list[MetricRow] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["run"] = run_path.name
            row["run_dir"] = str(run_path)
            rows.append(row)
    if not rows:
        raise ValueError(f"No metric rows found in {metrics_path}")
    return rows


def read_all_metrics(run_dirs: Iterable[str | Path]) -> list[MetricRow]:
    rows: list[MetricRow] = []
    for run_dir in run_dirs:
        rows.extend(read_metrics(run_dir))
    return rows


def final_rows(rows: Iterable[MetricRow]) -> list[MetricRow]:
    """Return latest row for each run/task/model/seed/phase/split group."""

    latest: dict[tuple[Any, ...], MetricRow] = {}
    for row in rows:
        key = _group_key(row)
        if key not in latest or int(row.get("step", -1)) >= int(latest[key].get("step", -1)):
            latest[key] = row
    return sorted(latest.values(), key=_sort_key)


def rows_at_steps(rows: Iterable[MetricRow], steps: Iterable[int]) -> list[MetricRow]:
    wanted = set(int(step) for step in steps)
    selected = [row for row in rows if int(row.get("step", -1)) in wanted]
    return sorted(selected, key=_sort_key)


def write_csv(rows: Iterable[MetricRow], path: str | Path, *, columns: list[str] | None = None) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or TABLE_COLUMNS
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_csv_value(row.get(column)) for column in columns})


def write_markdown(rows: Iterable[MetricRow], path: str | Path, *, columns: list[str] | None = None) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or TABLE_COLUMNS
    table_rows = list(rows)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in table_rows:
        lines.append("| " + " | ".join(_format_markdown_value(row.get(column)) for column in columns) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_steps(value: str | None) -> list[int]:
    if value is None or not value.strip():
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _group_key(row: MetricRow) -> tuple[Any, ...]:
    return (
        row.get("run"),
        row.get("task"),
        row.get("model"),
        row.get("seed"),
        row.get("phase"),
        row.get("split"),
    )


def _sort_key(row: MetricRow) -> tuple[str, str, int, str, str, int]:
    return (
        str(row.get("task") or ""),
        str(row.get("model") or ""),
        int(row.get("seed") or 0),
        str(row.get("phase") or ""),
        str(row.get("split") or ""),
        int(row.get("step") or 0),
    )


def _format_csv_value(value: Any) -> Any:
    return "" if value is None else value


def _format_markdown_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")

