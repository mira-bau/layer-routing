from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_initialization_sensitivity.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("initialization_sensitivity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spearman_uses_average_ranks_and_preserves_direction() -> None:
    module = _load_script()
    assert module._spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert module._spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert module._rank([1, 1, 3]) == [1.5, 1.5, 3.0]


def test_final_outcomes_accepts_unrounded_model_values(tmp_path: Path) -> None:
    module = _load_script()
    path = tmp_path / "outcomes.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seed", "baseline_l2", "baseline_l3", "saab_l2", "saab_l3"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "seed": 1001,
                "baseline_l2": 0.9042,
                "baseline_l3": 0.8991,
                "saab_l2": 0.8195,
                "saab_l3": 0.9479,
            }
        )

    outcomes = module._read_final_outcomes(path, (1001,))
    assert outcomes[1001]["delta_l2"] == pytest.approx(-0.0847)
    assert outcomes[1001]["delta_l3"] == pytest.approx(0.0488)
    assert outcomes[1001]["displacement_score"] == pytest.approx(0.1335)


def test_final_outcomes_requires_every_reported_seed(tmp_path: Path) -> None:
    module = _load_script()
    path = tmp_path / "outcomes.csv"
    path.write_text("seed,delta_l2,delta_l3\n42,-0.001,-0.034\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"missing seeds: \[1001\]"):
        module._read_final_outcomes(path, (42, 1001))
