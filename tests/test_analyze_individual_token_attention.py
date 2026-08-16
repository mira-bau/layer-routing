import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_individual_token_attention.py"
SPEC = importlib.util.spec_from_file_location("individual_token_attention", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _record(index, field_ids, attention_mask=None):
    if attention_mask is None:
        attention_mask = [1] * len(field_ids)
    return {
        "row_id": f"row_{index}",
        "input_ids": list(range(10, 10 + len(field_ids))),
        "field_ids": field_ids,
        "attention_mask": attention_mask,
        "tokens": [f"t{position}" for position in range(len(field_ids))],
    }


class IndividualTokenAttentionTests(unittest.TestCase):
    def test_eligibility_uses_valid_tokens_and_requires_each_named_field(self):
        record = _record(0, [4, 4, 4, 4, 3, 3, 3, 3, 0], [1] * 8 + [0])

        eligible, details = MODULE._eligibility(
            record,
            [3, 4],
            min_length=8,
            max_length=10,
            min_tokens_per_field=4,
            evaluation_max_length=256,
        )

        self.assertTrue(eligible)
        self.assertEqual(details["valid_token_count"], 8)
        self.assertEqual(details["named_field_token_counts"], {"3": 4, "4": 4})

    def test_selection_is_lowest_eligible_index_in_seeded_reservoir(self):
        records = []
        for index in range(30):
            if index == 25:
                fields = [4] * 4 + [3] * 21
            else:
                fields = [4] * 2 + [3] * 8
            records.append(_record(index, fields))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with mock.patch.object(MODULE, "EXPECTED_SOURCE_RECORDS", 30):
                selected, metadata = MODULE._select_candidate(
                    path,
                    [3, 4],
                    sample_size=30,
                    analysis_seed=1001,
                    min_length=20,
                    max_length=30,
                    min_tokens_per_field=4,
                    evaluation_max_length=256,
                )

        self.assertEqual(selected["source_index"], 25)
        self.assertFalse(
            metadata["selection_rule"]["model_outputs_used_for_selection"]
        )

    def test_same_field_mass_and_boundaries(self):
        matrix = [
            [0.6, 0.2, 0.1, 0.1],
            [0.3, 0.5, 0.1, 0.1],
            [0.1, 0.1, 0.5, 0.3],
            [0.1, 0.1, 0.2, 0.6],
        ]
        field_ids = [4, 4, 3, 3]

        self.assertAlmostEqual(MODULE._same_field_mass(matrix, field_ids), 0.8)
        self.assertEqual(MODULE._field_boundaries(field_ids), [2])

    def test_top_changes_have_deterministic_tie_breaks(self):
        baseline = [[0.5, 0.5], [0.5, 0.5]]
        saab = [[0.6, 0.4], [0.4, 0.6]]
        rows = MODULE._top_changes(
            baseline,
            saab,
            ["a", "b"],
            [4, 3],
            {3: "content", 4: "title"},
            layer=2,
            limit=2,
        )

        self.assertEqual(
            [(row["query_position"], row["key_position"]) for row in rows],
            [(0, 0), (0, 1)],
        )


if __name__ == "__main__":
    unittest.main()
