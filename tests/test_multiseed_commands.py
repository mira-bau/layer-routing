from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pytest

from dbpedia.scripts.multiseed_commands import (
    DEFAULT_MODELS,
    DEFAULT_SEEDS,
    main,
    table_command,
    train_command,
)


def test_paper_defaults_cover_eight_seeds_and_two_models() -> None:
    assert DEFAULT_SEEDS == "0,7,42,99,123,256,1001,2024"
    assert DEFAULT_MODELS == "baseline,saab"


def test_train_command_uses_shared_recipe_model_seed_and_paths() -> None:
    command = train_command(
        python="python",
        env_prefix="PYTHONPATH=src",
        config=Path("dbpedia/configs/msm_dbpedia_full_recipe.yaml"),
        model="saab",
        seed=123,
        train_jsonl=Path("data/train.jsonl"),
        val_jsonl=Path("data/val.jsonl"),
        tokenizer_json=Path("data/tokenizer.json"),
        run_root=Path("runs/dbpedia_multiseed"),
    )

    assert command.startswith(
        "PYTHONPATH=src python -m structformer.training.train_msm"
    )
    assert "--config dbpedia/configs/msm_dbpedia_full_recipe.yaml" in command
    assert "--model saab" in command
    assert "--seed 123" in command
    assert "--train-jsonl data/train.jsonl" in command
    assert "--val-jsonl data/val.jsonl" in command
    assert "--tokenizer-json data/tokenizer.json" in command
    assert "--run-dir runs/dbpedia_multiseed/saab_seed123" in command


def test_table_command_scans_multiseed_run_root() -> None:
    command = table_command(
        python="python",
        env_prefix="PYTHONPATH=src",
        run_root=Path("runs/dbpedia_multiseed"),
        table_out_dir=Path("outputs/tables/dbpedia_multiseed"),
        steps="1,10,50,100,200,300,500",
    )

    assert "scripts/make_tables.py" in command
    assert "--runs runs/dbpedia_multiseed" in command
    assert "--out-dir outputs/tables/dbpedia_multiseed" in command
    assert "--steps 1,10,50,100,200,300,500" in command


def test_cli_prints_sixteen_paper_commands_and_table_command() -> None:
    stdout = StringIO()
    with redirect_stdout(stdout):
        assert (
            main(
                [
                    "--train-jsonl",
                    "data/train.jsonl",
                    "--val-jsonl",
                    "data/val.jsonl",
                    "--tokenizer-json",
                    "data/tokenizer.json",
                ]
            )
            == 0
        )

    output = stdout.getvalue()
    assert "# DBpedia eight-seed train commands: 16 runs" in output
    assert output.count("python -m structformer.training.train_msm") == 16
    assert "--run-dir runs/dbpedia_multiseed/baseline_seed0" in output
    assert "--run-dir runs/dbpedia_multiseed/saab_seed2024" in output
    assert "# Table command, after all runs finish" in output


def test_cli_rejects_unreported_model() -> None:
    with pytest.raises(SystemExit, match="models contains invalid values: other"):
        main(
            [
                "--train-jsonl",
                "data/train.jsonl",
                "--val-jsonl",
                "data/val.jsonl",
                "--tokenizer-json",
                "data/tokenizer.json",
                "--models",
                "baseline,other",
            ]
        )
