import importlib.util
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch

    from structformer.models.bias import SAABWeights, build_saab_bias


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class SAABBiasTests(unittest.TestCase):
    def test_field_only_bias(self):
        field_ids = torch.tensor([[3, 4, 3]])
        bias = build_saab_bias(field_ids, weights=SAABWeights(field=1.0, entity=0.0, value_type=0.0, time=0.0))

        expected = torch.tensor(
            [
                [
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                ]
            ]
        )
        self.assertTrue(torch.equal(bias, expected))

    def test_combines_available_tags(self):
        field_ids = torch.tensor([[1, 2]])
        entity_ids = torch.tensor([[7, 7]])
        value_type_ids = torch.tensor([[3, 4]])
        bias = build_saab_bias(
            field_ids,
            entity_ids=entity_ids,
            value_type_ids=value_type_ids,
            weights=SAABWeights(field=1.0, entity=2.0, value_type=0.5, time=0.0),
        )

        expected = torch.tensor([[[3.5, 2.0], [2.0, 3.5]]])
        self.assertTrue(torch.equal(bias, expected))


if __name__ == "__main__":
    unittest.main()

