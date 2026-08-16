import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_opportunity_normalized_sfm.py"
SPEC = importlib.util.spec_from_file_location("opportunity_sfm", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class OpportunityNormalizedSFMTests(unittest.TestCase):
    def test_balanced_two_field_reference_is_one_half(self):
        value = MODULE.opportunity_reference(
            [3, 3, 4, 4], [1, 1, 1, 1], {3, 4}, 4
        )
        self.assertAlmostEqual(value, 0.5)

    def test_unbalanced_field_reference_uses_available_pairs(self):
        value = MODULE.opportunity_reference(
            [3, 3, 3, 4], [1, 1, 1, 1], {3, 4}, 4
        )
        self.assertAlmostEqual(value, 10 / 16)

    def test_padding_unknown_fields_and_truncation_are_excluded(self):
        value = MODULE.opportunity_reference(
            [3, 3, 4, 0, 5], [1, 1, 1, 0, 1], {3, 4}, 4
        )
        self.assertAlmostEqual(value, 5 / 9)

    def test_adjustment_has_neutral_and_all_same_anchors(self):
        opportunity = 0.6
        self.assertAlmostEqual(MODULE.adjusted_sfm(opportunity, opportunity), 0.0)
        self.assertAlmostEqual(MODULE.adjusted_sfm(1.0, opportunity), 1.0)
        self.assertLess(MODULE.adjusted_sfm(0.4, opportunity), 0.0)

    def test_adjustment_is_undefined_without_cross_field_opportunity(self):
        self.assertIsNone(MODULE.adjusted_sfm(1.0, 1.0))

    def test_holm_adjustment_preserves_original_order(self):
        adjusted = MODULE._holm([0.04, 0.01, 0.03, 0.2])
        self.assertEqual(adjusted, [0.09, 0.04, 0.09, 0.2])


if __name__ == "__main__":
    unittest.main()
