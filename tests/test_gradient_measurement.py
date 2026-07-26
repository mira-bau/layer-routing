from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_analyzer():
    path = REPO_ROOT / "scripts/analyze_training_gradients.py"
    spec = importlib.util.spec_from_file_location("gradient_analyzer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_prepared_jsonl(path: Path) -> None:
    records = [
        {
            "row_id": f"row_{index}",
            "label": None,
            "input_ids": [2 + index, 6 + index, 10 + index, 14 + index],
            "field_ids": [3, 3, 4, 4],
            "attention_mask": [1, 1, 1, 1],
            "tokens": ["a", "b", "c", "d"],
        }
        for index in range(4)
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_training_logs_true_msm_gradient_metadata(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from structformer.training.train_msm import run_train_msm

    train_jsonl = tmp_path / "train.jsonl"
    _write_prepared_jsonl(train_jsonl)
    run_root = tmp_path / "runs"
    shared_config = {
        "seed": 7,
        "device": "cpu",
        "allow_cpu": True,
        "data": {
            "train_jsonl": str(train_jsonl),
            "max_length": 8,
            "field_vocab_size": 6,
            "mask_field_id": 5,
            "num_labels": 5,
            "mask_probability": 0.5,
        },
        "model_config": {
            "d_model": 12,
            "num_layers": 2,
            "num_heads": 3,
            "ff_dim": 24,
            "dropout": 0.0,
        },
        "training": {
            "max_steps": 2,
            "microbatch_size": 1,
            "gradient_accumulation_steps": 2,
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "log_every_steps": 1,
            "eval_every_steps": 0,
            "checkpoint_every_steps": 0,
            "checkpoint_diagnostic_steps": [1, 2],
            "log_layer_gradients": True,
        },
    }
    for model in ("baseline", "saab"):
        run_train_msm(
            {"model": model, **shared_config},
            run_dir=run_root / f"{model}_seed7",
        )

    gradient_path = run_root / "saab_seed7/layer_gradients/metrics.jsonl"
    rows = [
        json.loads(line)
        for line in gradient_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    for row in rows:
        assert row["objective"] == "msm_cross_entropy"
        assert (
            row["measurement_point"]
            == "after_gradient_accumulation_before_global_clipping"
        )
        assert len(row["qkv_grad_norm_per_layer"]) == 2
        assert row["grad_norm_ratio_last_penultimate"] is not None

    diagnostics_out = tmp_path / "timeseries"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/diag_timeseries.py"),
            "--run-root",
            str(run_root),
            "--seed",
            "7",
            "--val-jsonl",
            str(train_jsonl),
            "--n-examples",
            "2",
            "--out-dir",
            str(diagnostics_out),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    timeseries = json.loads(
        (diagnostics_out / "timeseries_diagnostics.json").read_text(
            encoding="utf-8"
        )
    )
    assert timeseries["steps"] == [1, 2]
    assert set(timeseries["by_step"]["2"]) == {"baseline", "saab"}


def test_analyzer_rejects_non_msm_gradient_logs(tmp_path: Path) -> None:
    analyzer = _load_analyzer()
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps(
            {
                "model": "saab",
                "seed": 1001,
                "step": 1,
                "objective": "logits_sum",
                "measurement_point": (
                    "after_gradient_accumulation_before_global_clipping"
                ),
                "qkv_grad_norm_per_layer": [1.0, 2.0],
                "grad_norm_ratio_last_penultimate": 2.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="objective"):
        analyzer._load_gradient_rows(
            path,
            expected_model="saab",
            expected_seed=1001,
        )
