"""Run logging helpers with plain, inspectable artifacts."""

from __future__ import annotations

import csv
import json
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class RunLogger:
    """Write live progress and persistent metrics for a run."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_jsonl = self.run_dir / "metrics.jsonl"
        self.metrics_csv = self.run_dir / "metrics.csv"

    def write_json(self, name: str, data: Mapping[str, Any]) -> Path:
        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    def write_text(self, name: str, text: str) -> Path:
        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def log_metric(self, row: Mapping[str, Any]) -> None:
        enriched = {"time": utc_now_iso(), **dict(row)}
        with self.metrics_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(enriched, sort_keys=True) + "\n")
        self._append_csv(enriched)

    def progress(self, phase: str, **fields: Any) -> None:
        parts = [f"[{phase}]"]
        for key, value in fields.items():
            if value is None:
                continue
            parts.append(f"{key}={value}")
        print(" ".join(parts), file=sys.stdout, flush=True)

    def _append_csv(self, row: Mapping[str, Any]) -> None:
        fieldnames = list(row.keys())
        write_header = not self.metrics_csv.exists()
        with self.metrics_csv.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(dict(row))

