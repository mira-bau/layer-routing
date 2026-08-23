from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.extract_final_sfm_outcomes import build_rows, main


def _write_diagnostic(root: Path, seed: int, step: int = 500) -> None:
    directory = root / f"seed{seed}"
    directory.mkdir(parents=True)
    payload = {
        "baseline": {
            "seed": seed,
            "step": step,
            "same_field_mass_per_layer": [0.7, 0.8, 0.9, 0.85],
        },
        "saab": {
            "seed": seed,
            "step": step,
            "same_field_mass_per_layer": [0.75, 0.78, 0.82, 0.95],
        },
    }
    (directory / "attention_diagnostics.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_builds_unrounded_layer_outcomes(tmp_path: Path) -> None:
    _write_diagnostic(tmp_path, 42)

    rows = build_rows(tmp_path, [42], expected_step=500)

    assert rows[0]["baseline_l2"] == 0.9
    assert rows[0]["saab_l3"] == 0.95
    assert rows[0]["delta_l2"] == pytest.approx(-0.08)
    assert rows[0]["delta_l3"] == pytest.approx(0.10)


def test_cli_writes_initialization_input_csv(tmp_path: Path) -> None:
    _write_diagnostic(tmp_path, 7)
    output = tmp_path / "final_sfm_by_seed.csv"

    assert main(
        [
            "--diagnostics-root",
            str(tmp_path),
            "--seeds",
            "7",
            "--output",
            str(output),
        ]
    ) == 0

    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert rows[0]["seed"] == "7"
    assert rows[0]["delta_l2"] == str(0.82 - 0.9)


def test_rejects_wrong_checkpoint_step(tmp_path: Path) -> None:
    _write_diagnostic(tmp_path, 99, step=300)

    with pytest.raises(ValueError, match="expected 500"):
        build_rows(tmp_path, [99], expected_step=500)
