from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_paired_sfm_length import DatasetSpec, _verify_checkpoint
from scripts.diag_attention import _load_real_examples


def test_attention_diagnostic_uses_prepared_attention_mask(tmp_path: Path) -> None:
    path = tmp_path / "val.jsonl"
    path.write_text(
        json.dumps(
            {
                "input_ids": [7, 0, 8],
                "field_ids": [4, 4, 3],
                "attention_mask": [1, 1, 0],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    batch = _load_real_examples(path, 1)

    assert batch["attention_mask"].tolist() == [[True, True, False]]


def _checkpoint_config() -> dict:
    return {
        "step": 500,
        "config": {
            "model": "saab",
            "seed": 1001,
            "data": {
                "max_length": 256,
                "num_labels": 5,
                "field_vocab_size": 6,
                "mask_probability": 0.15,
            },
            "model_config": {
                "d_model": 768,
                "num_layers": 4,
                "num_heads": 6,
                "ff_dim": 3072,
                "dropout": 0.2,
                "scale_embeddings": True,
                "saab_field_weight": 1.0,
                "saab_layer_mask": [],
                "saab_shuffle_bias": False,
            },
            "training": {
                "max_steps": 500,
                "microbatch_size": 64,
                "gradient_accumulation_steps": 8,
                "learning_rate": 1.0e-4,
                "lr_schedule": "linear_warmup_cosine",
                "warmup_steps": 50,
                "min_lr_ratio": 0.1,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "grad_clip": 1.0,
            },
        },
    }


def test_paired_analysis_rejects_nonpaper_optimizer_setting(tmp_path: Path) -> None:
    spec = DatasetSpec(
        name="DBpedia",
        baseline_checkpoint=tmp_path / "baseline.pt",
        saab_checkpoint=tmp_path / "saab.pt",
        validation_jsonl=tmp_path / "val.jsonl",
        field_vocab_json=tmp_path / "field_vocab.json",
        expected_step=500,
        expected_max_length=256,
        expected_num_labels=5,
        expected_field_vocab_size=6,
    )
    checkpoint = _checkpoint_config()
    checkpoint["config"]["training"]["weight_decay"] = 0.0

    with pytest.raises(ValueError, match="weight_decay"):
        _verify_checkpoint(checkpoint, spec, "saab")
