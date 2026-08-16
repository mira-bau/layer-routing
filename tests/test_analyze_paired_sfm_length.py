import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch

    from scripts.analyze_paired_sfm_length import (
        _holm_adjust,
        _length_stats,
        _per_example_sfm,
        _select_records,
        _write_csv,
    )


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class PairedSFMLengthAnalysisTests(unittest.TestCase):
    def test_per_example_sfm_preserves_example_axis(self):
        field_ids = torch.tensor([[3, 3, 4, 4], [3, 4, 0, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
        uniform = torch.full((2, 1, 4, 4), 0.25)
        uniform[1, :, :, 2:] = 0.0
        uniform[1, :, :, :2] = 0.5

        values = _per_example_sfm([uniform], field_ids, attention_mask, [3, 4])

        self.assertEqual(tuple(values.shape), (2, 1))
        self.assertTrue(torch.allclose(values[:, 0], torch.tensor([0.5, 0.5])))

    def test_holm_adjustment_is_monotone_in_sorted_order(self):
        adjusted = _holm_adjust([0.01, 0.04, 0.03, 0.002])

        self.assertEqual(adjusted, [0.03, 0.06, 0.06, 0.008])

    def test_record_selection_keeps_first_records_and_fixed_reservoir(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "val.jsonl"
            with path.open("w") as handle:
                for index in range(10):
                    record = {
                        "row_id": f"row_{index}",
                        "input_ids": [2, index + 3],
                        "field_ids": [3, 4],
                        "attention_mask": [1, 1],
                    }
                    handle.write(json.dumps(record) + "\n")

            selected_a, primary_a, length_a, count_a = _select_records(
                path, primary_size=3, length_sample_size=5, sample_seed=1001
            )
            selected_b, primary_b, length_b, count_b = _select_records(
                path, primary_size=3, length_sample_size=5, sample_seed=1001
            )

        self.assertEqual(primary_a, {0, 1, 2})
        self.assertEqual(primary_a, primary_b)
        self.assertEqual(length_a, length_b)
        self.assertEqual(len(length_a), 5)
        self.assertEqual(count_a, count_b)
        self.assertEqual(
            [record["source_index"] for record in selected_a],
            [record["source_index"] for record in selected_b],
        )

    def test_length_quartiles_remain_nonempty_when_cutpoints_are_tied(self):
        lengths = list(range(1, 11)) + [256] * 30
        rows = []
        for index, length in enumerate(lengths):
            row = {"source_index": index, "evaluation_token_length": length}
            for layer in range(4):
                row[f"delta_l{layer}"] = index / 100 + layer / 1000
            rows.append(row)
        args = SimpleNamespace(analysis_seed=1001, correlation_bootstrap_resamples=20)

        _, bins = _length_stats(rows, args, "tied")

        self.assertEqual({row["length_quartile_bin"] for row in rows}, {1, 2, 3, 4})
        self.assertTrue(all(row["examples"] == 10 for row in bins))

    def test_csv_accepts_fields_added_to_only_some_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.csv"
            _write_csv(path, [{"a": 1}, {"a": 2, "quartile": 3}])
            contents = path.read_text()

        self.assertIn("a,quartile", contents)
        self.assertIn("2,3", contents)


if __name__ == "__main__":
    unittest.main()
