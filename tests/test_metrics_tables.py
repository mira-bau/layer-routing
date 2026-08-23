import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts.make_tables import main as make_tables_main
from structformer.analysis.metrics import discover_run_dirs, final_rows, parse_steps, read_all_metrics, rows_at_steps


class MetricsTableTests(unittest.TestCase):
    def test_discovers_and_summarizes_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_metrics(
                root / "baseline",
                [
                    {"task": "msm", "model": "baseline", "seed": 42, "phase": "train", "split": "train", "step": 1, "loss": 1.4},
                    {"task": "msm", "model": "baseline", "seed": 42, "phase": "train", "split": "train", "step": 2, "loss": 1.2},
                    {"task": "msm", "model": "baseline", "seed": 42, "phase": "eval", "split": "val", "step": 2, "loss": 1.1, "accuracy": 0.5},
                ],
            )
            _write_metrics(
                root / "saab",
                [
                    {"task": "msm", "model": "saab", "seed": 42, "phase": "train", "split": "train", "step": 2, "loss": 1.0},
                ],
            )

            run_dirs = discover_run_dirs([root])
            self.assertEqual([path.name for path in run_dirs], ["baseline", "saab"])

            rows = read_all_metrics(run_dirs)
            final = final_rows(rows)
            self.assertEqual(len(final), 3)
            baseline_train = [row for row in final if row["model"] == "baseline" and row["phase"] == "train"][0]
            self.assertEqual(baseline_train["step"], 2)
            self.assertEqual(baseline_train["loss"], 1.2)

            step_rows = rows_at_steps(rows, [1])
            self.assertEqual(len(step_rows), 1)
            self.assertEqual(step_rows[0]["step"], 1)

    def test_parse_steps(self):
        self.assertEqual(parse_steps("10, 50,100"), [10, 50, 100])
        self.assertEqual(parse_steps(None), [])

    def test_make_tables_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_metrics(
                root / "run",
                [
                    {"task": "msm", "model": "saab", "seed": 42, "phase": "train", "split": "train", "step": 10, "loss": 0.9},
                    {"task": "msm", "model": "saab", "seed": 42, "phase": "eval", "split": "val", "step": 10, "loss": 0.8, "accuracy": 0.7},
                ],
            )
            out_dir = root / "tables"

            with redirect_stdout(StringIO()):
                self.assertEqual(make_tables_main(["--runs", str(root), "--out-dir", str(out_dir), "--steps", "10"]), 0)

            self.assertTrue((out_dir / "final_metrics.csv").exists())
            self.assertTrue((out_dir / "final_metrics.md").exists())
            self.assertTrue((out_dir / "step_metrics.csv").exists())
            self.assertIn("saab", (out_dir / "final_metrics.md").read_text(encoding="utf-8"))


def _write_metrics(run_dir: Path, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    unittest.main()
