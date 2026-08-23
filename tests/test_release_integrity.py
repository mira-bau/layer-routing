from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(relative_path: str, module_name: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_contains_data_loaders() -> None:
    assert (REPO_ROOT / "src/structformer/data/dbpedia.py").is_file()
    assert (REPO_ROOT / "src/structformer/data/msm_jsonl.py").is_file()


def test_release_contains_revised_manuscript_analysis_paths() -> None:
    required = (
        "dbpedia/scripts/prepare_external_dbpedia_split.py",
        "scripts/analyze_initialization_sensitivity.py",
        "scripts/analyze_individual_token_attention.py",
        "scripts/analyze_opportunity_normalized_sfm.py",
        "scripts/analyze_paired_sfm_length.py",
        "scripts/analyze_untouched_dbpedia_test.py",
        "scripts/benchmark_computational_overhead.py",
    )
    for relative_path in required:
        assert (REPO_ROOT / relative_path).is_file(), relative_path


def test_release_documents_reported_environment() -> None:
    assert (REPO_ROOT / "PAPER_ENVIRONMENT.md").is_file()
    assert (REPO_ROOT / "requirements-paper.txt").is_file()


def test_baseline_and_saab_have_equal_parameter_counts() -> None:
    torch = pytest.importorskip("torch")
    from structformer.models import StructuredTransformerModel, TransformerConfig

    shared = dict(
        vocab_size=64,
        field_vocab_size=6,
        max_length=16,
        head_type="token",
        num_labels=5,
        d_model=24,
        num_layers=2,
        num_heads=3,
        ff_dim=48,
        dropout=0.0,
    )
    baseline = StructuredTransformerModel(
        TransformerConfig(variant="baseline", **shared)
    )
    saab = StructuredTransformerModel(TransformerConfig(variant="saab", **shared))
    assert baseline.parameter_count() == saab.parameter_count()
    assert sum(p.numel() for p in baseline.parameters()) == sum(
        p.numel() for p in saab.parameters()
    )
    del torch


def test_paper_configs_share_training_recipe() -> None:
    yaml = pytest.importorskip("yaml")
    config_paths = [
        REPO_ROOT / "dbpedia/configs/msm_baseline.yaml",
        REPO_ROOT / "dbpedia/configs/msm_saab.yaml",
        REPO_ROOT / "dbpedia/configs/msm_dbpedia_full_recipe.yaml",
    ]
    configs = [
        yaml.safe_load(path.read_text(encoding="utf-8")) for path in config_paths
    ]
    keys = (
        "max_steps",
        "microbatch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "lr_schedule",
        "warmup_steps",
        "min_lr_ratio",
        "weight_decay",
        "grad_clip",
    )
    reference = {key: configs[0]["training"][key] for key in keys}
    for config in configs[1:]:
        assert {key: config["training"][key] for key in keys} == reference
    assert reference["max_steps"] == 500
    assert reference["microbatch_size"] * reference[
        "gradient_accumulation_steps"
    ] == 512
    assert all(config["model_config"]["scale_embeddings"] for config in configs)


def test_pubmed_configs_use_reported_step_count() -> None:
    yaml = pytest.importorskip("yaml")
    for path in (REPO_ROOT / "pubmed/configs").glob("*.yaml"):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["training"]["max_steps"] == 1500
        assert config["training"]["microbatch_size"] == 64
        assert config["training"]["gradient_accumulation_steps"] == 8
        assert config["training"]["eval_every_steps"] == 500
        assert config["training"]["checkpoint_every_steps"] == 500
        assert config["model_config"]["scale_embeddings"] is True


def test_multiseed_generator_defaults_match_paper() -> None:
    module = _load_script(
        "dbpedia/scripts/multiseed_commands.py",
        "multiseed_commands",
    )
    assert module.DEFAULT_SEEDS == "0,7,42,99,123,256,1001,2024"
    assert module.DEFAULT_MODELS == "baseline,saab"


def test_attention_synthetic_batch_has_detectable_fields() -> None:
    module = _load_script("scripts/diag_attention.py", "diag_attention")
    batch = module._make_synthetic_batch(n=2, seq_len=8, vocab_size=64)
    fields = module._detect_named_fields(batch, mask_field_id=5)
    assert fields == {3: "field_3", 4: "field_4"}
    assert module._valid_mask(
        batch["field_ids"],
        batch["attention_mask"],
        fields,
    ).all()
