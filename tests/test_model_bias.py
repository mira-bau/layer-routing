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
        bias = build_saab_bias(field_ids, weights=SAABWeights(field=1.0))

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


if __name__ == "__main__":
    unittest.main()
