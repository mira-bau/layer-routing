import importlib.util
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch

    from structformer.tasks.msm import (
        MSM_IGNORE_INDEX,
        make_synthetic_msm_batch,
        mask_field_ids,
        msm_cross_entropy,
    )


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class MSMTaskTests(unittest.TestCase):
    def test_mask_field_ids_keeps_tokens_implicit_and_labels_only_masked(self):
        field_ids = torch.tensor([[4, 4, 4, 3, 3, 3]])
        attention_mask = torch.ones_like(field_ids, dtype=torch.bool)
        generator = torch.Generator()
        generator.manual_seed(7)

        masked, labels, positions = mask_field_ids(
            field_ids,
            attention_mask,
            mask_field_id=5,
            mask_probability=0.34,
            generator=generator,
        )

        self.assertEqual(int(positions.sum().item()), 2)
        self.assertEqual(int((positions & field_ids.eq(4)).sum().item()), 1)
        self.assertEqual(int((positions & field_ids.eq(3)).sum().item()), 1)
        self.assertTrue(torch.equal(masked[positions], torch.full((2,), 5)))
        self.assertTrue(torch.equal(labels[positions], field_ids[positions]))
        self.assertTrue(torch.equal(labels[~positions], torch.full_like(labels[~positions], MSM_IGNORE_INDEX)))

    def test_synthetic_batch_has_field_token_correlation(self):
        batch = make_synthetic_msm_batch(batch_size=2, seq_len=6, vocab_size=20)

        self.assertEqual(tuple(batch.input_ids.shape), (2, 6))
        self.assertTrue(torch.all(batch.field_ids[:, :3].eq(4)))
        self.assertTrue(torch.all(batch.field_ids[:, 3:].eq(3)))
        self.assertTrue(torch.all(batch.input_ids[:, :3] < 10))
        self.assertTrue(torch.all(batch.input_ids[:, 3:] >= 10))

    def test_msm_cross_entropy(self):
        logits = torch.randn(1, 3, 5)
        labels = torch.tensor([[MSM_IGNORE_INDEX, 4, 3]])

        loss = msm_cross_entropy(logits, labels)

        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()

