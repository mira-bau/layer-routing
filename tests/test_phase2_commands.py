import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from dbpedia.scripts.phase2_dbpedia_commands import main, table_command, train_command


class Phase2CommandTests(unittest.TestCase):
    def test_train_command_uses_model_seed_and_paths(self):
        command = train_command(
            python="python",
            env_prefix="PYTHONPATH=src",
            config=Path("configs/msm_dbpedia_full_recipe.yaml"),
            model="saab",
            seed=123,
            train_jsonl=Path("data/train.jsonl"),
            val_jsonl=Path("data/val.jsonl"),
            tokenizer_json=Path("data/tokenizer.json"),
            run_root=Path("runs/phase2"),
        )

        self.assertIn("PYTHONPATH=src python -m structformer.training.train_msm", command)
        self.assertIn("--model saab", command)
        self.assertIn("--seed 123", command)
        self.assertIn("--run-dir runs/phase2/saab_seed123", command)

    def test_table_command_scans_run_root(self):
        command = table_command(
            python="python",
            env_prefix="PYTHONPATH=src",
            run_root=Path("runs/phase2"),
            table_out_dir=Path("outputs/phase2"),
            steps="100,500",
        )

        self.assertIn("scripts/make_tables.py", command)
        self.assertIn("--runs runs/phase2", command)
        self.assertIn("--steps 100,500", command)

    def test_cli_prints_separate_commands(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(
                main(
                    [
                        "--train-jsonl",
                        "data/train.jsonl",
                        "--val-jsonl",
                        "data/val.jsonl",
                        "--tokenizer-json",
                        "data/tokenizer.json",
                        "--seeds",
                        "1,2",
                        "--models",
                        "baseline,casa",
                        "--no-table",
                    ]
                ),
                0,
            )

        output = stdout.getvalue()
        self.assertIn("# DBpedia Phase 2 train commands: 4 runs", output)
        self.assertIn("--run-dir runs/dbpedia_phase2_full/baseline_seed1", output)
        self.assertIn("--run-dir runs/dbpedia_phase2_full/casa_seed2", output)


if __name__ == "__main__":
    unittest.main()
