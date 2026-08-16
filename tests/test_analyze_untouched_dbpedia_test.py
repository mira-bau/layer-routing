import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_untouched_dbpedia_test.py"
SPEC = importlib.util.spec_from_file_location("untouched_dbpedia", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class UntouchedDBpediaTestAnalysisTests(unittest.TestCase):
    def test_rows_include_raw_and_opportunity_adjusted_values(self):
        records = [
            {
                "source_index": 7,
                "row_id": "row_7",
                "field_ids": [3, 3, 4, 4],
                "attention_mask": [1, 1, 1, 1],
            }
        ]
        baseline = {7: [0.6, 0.7, 0.8, 0.9]}
        saab = {7: [0.7, 0.6, 0.75, 0.95]}

        row = MODULE._make_rows(records, baseline, saab, [3, 4], 256)[0]

        self.assertAlmostEqual(row["opportunity_reference"], 0.5)
        self.assertAlmostEqual(row["delta_l0"], 0.1)
        self.assertAlmostEqual(row["baseline_adjusted_l0"], 0.2)
        self.assertAlmostEqual(row["saab_adjusted_l0"], 0.4)
        self.assertAlmostEqual(row["delta_adjusted_l0"], 0.2)

    def test_undefined_adjusted_rows_are_excluded_from_adjusted_stats(self):
        rows = [{"delta_adjusted_l0": None}, {"delta_adjusted_l0": 0.1}]
        for layer in range(4):
            rows[1][f"baseline_adjusted_l{layer}"] = 0.2
            rows[1][f"saab_adjusted_l{layer}"] = 0.3
            rows[1][f"delta_adjusted_l{layer}"] = 0.1

        adjusted = MODULE._adjusted_stats_rows(rows)

        self.assertEqual(len(adjusted), 1)
        self.assertEqual(adjusted[0]["baseline_l2"], 0.2)
        self.assertEqual(adjusted[0]["saab_l2"], 0.3)


if __name__ == "__main__":
    unittest.main()
