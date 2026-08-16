import importlib.util
import io
import json
import tempfile
import unittest
from copy import deepcopy
from contextlib import redirect_stdout
from pathlib import Path


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TOKENIZERS = importlib.util.find_spec("tokenizers") is not None

if HAS_TORCH and HAS_TOKENIZERS:
    from dbpedia.scripts.prepare_dbpedia import main as prepare_dbpedia_main
    from structformer.training.train_msm import run_train_msm


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "dbpedia"


@unittest.skipUnless(HAS_TORCH and HAS_TOKENIZERS, "PyTorch and tokenizers are required")
class TrainMSMTests(unittest.TestCase):
    def test_runs_two_steps_on_prepared_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            processed = tmp_path / "processed"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    prepare_dbpedia_main(
                        [
                            "--train-csv",
                            str(FIXTURE_DIR / "train.csv"),
                            "--val-csv",
                            str(FIXTURE_DIR / "val.csv"),
                            "--out-dir",
                            str(processed),
                            "--vocab-size",
                            "80",
                            "--max-length",
                            "16",
                        ]
                    ),
                    0,
                )

            run_dir = tmp_path / "run"
            config = {
                "model": "baseline",
                "allow_cpu": True,
                "data": {
                    "train_jsonl": str(processed / "train.jsonl"),
                    "val_jsonl": str(processed / "val.jsonl"),
                    "tokenizer_json": str(processed / "tokenizer.json"),
                    "train_sample_size": 3,
                    "val_sample_size": 2,
                    "sample_seed": 7,
                    "max_length": 16,
                },
                "model_config": {
                    "d_model": 16,
                    "num_layers": 1,
                    "num_heads": 4,
                    "ff_dim": 32,
                    "dropout": 0.0,
                    "casa_rank": 4,
                    "scale_embeddings": True,
                },
                "training": {
                    "max_steps": 2,
                    "microbatch_size": 2,
                    "gradient_accumulation_steps": 2,
                    "learning_rate": 0.001,
                    "lr_schedule": "linear_warmup_cosine",
                    "warmup_steps": 1,
                    "min_lr_ratio": 0.5,
                    "weight_decay": 0.0,
                    "log_every_steps": 1,
                    "eval_every_steps": 1,
                    "checkpoint_every_steps": 0,
                    "log_layer_gradients": True,
                    "num_workers": 0,
                },
            }
            with redirect_stdout(io.StringIO()):
                run_train_msm(config, run_dir=run_dir)

            self.assertTrue((run_dir / "metrics.jsonl").exists())
            self.assertTrue((run_dir / "sample_batch.json").exists())
            self.assertTrue((run_dir / "sample_rows.json").exists())
            self.assertTrue((run_dir / "data_manifest.json").exists())
            self.assertTrue((run_dir / "final_summary.json").exists())
            self.assertTrue((run_dir / "layer_gradients" / "metrics.jsonl").exists())
            manifest = json.loads((run_dir / "data_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["train_records"], 3)
            self.assertGreaterEqual(manifest["train_source_records"], 2)
            self.assertEqual(manifest["train_sample_size"], 3)
            self.assertEqual(manifest["train_sample_seed"], 7)
            self.assertIsNotNone(manifest["train_sample_indices_hash"])
            self.assertEqual(manifest["val_records"], 2)
            self.assertEqual(manifest["val_sample_size"], 2)
            self.assertEqual(manifest["val_sample_seed"], 7)
            rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertGreaterEqual(len(rows), 3)
            self.assertEqual(rows[0]["split"], "train")
            self.assertIn("loss", rows[0])
            train_rows = [row for row in rows if row["phase"] == "train"]
            self.assertAlmostEqual(train_rows[0]["lr"], 0.0005)
            self.assertAlmostEqual(train_rows[1]["lr"], 0.001)
            summary = json.loads((run_dir / "final_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["steps"], 2)
            self.assertEqual(summary["gradient_accumulation_steps"], 2)
            self.assertEqual(summary["effective_batch_size"], 4)
            self.assertTrue(summary["layer_gradient_logging"])
            self.assertEqual(summary["final_eval"]["step"], 2)
            self.assertIn("loss", summary["final_eval"])
            self.assertIn("accuracy", summary["final_eval"])
            gradient_rows = [
                json.loads(line)
                for line in (run_dir / "layer_gradients" / "metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(gradient_rows), 2)
            self.assertEqual(gradient_rows[0]["objective"], "msm_cross_entropy")
            self.assertEqual(
                gradient_rows[0]["measurement_point"],
                "after_gradient_accumulation_before_global_clipping",
            )
            self.assertEqual(len(gradient_rows[0]["qkv_grad_norm_per_layer"]), 1)
            self.assertGreater(gradient_rows[0]["qkv_grad_norm_per_layer"][0], 0.0)

            saab_run_dir = tmp_path / "run_saab"
            saab_config = deepcopy(config)
            saab_config["model"] = "saab"
            with redirect_stdout(io.StringIO()):
                run_train_msm(saab_config, run_dir=saab_run_dir)

            baseline_rows = json.loads((run_dir / "sample_rows.json").read_text(encoding="utf-8"))
            saab_rows = json.loads((saab_run_dir / "sample_rows.json").read_text(encoding="utf-8"))
            baseline_batch = json.loads((run_dir / "sample_batch.json").read_text(encoding="utf-8"))
            saab_batch = json.loads((saab_run_dir / "sample_batch.json").read_text(encoding="utf-8"))
            baseline_model = json.loads((run_dir / "model_summary.json").read_text(encoding="utf-8"))
            saab_model = json.loads((saab_run_dir / "model_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(baseline_rows, saab_rows)
            self.assertEqual(baseline_batch, saab_batch)
            self.assertEqual(baseline_model, saab_model)

            casa_run_dir = tmp_path / "run_casa"
            casa_config = deepcopy(config)
            casa_config["model"] = "casa"
            with redirect_stdout(io.StringIO()):
                run_train_msm(casa_config, run_dir=casa_run_dir)

            casa_rows = json.loads((casa_run_dir / "sample_rows.json").read_text(encoding="utf-8"))
            self.assertEqual(baseline_rows["row_ids"], casa_rows["row_ids"])


if __name__ == "__main__":
    unittest.main()
