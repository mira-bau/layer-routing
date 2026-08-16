import importlib.util
import math
import unittest
from dataclasses import replace
from unittest.mock import patch


HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch

    from structformer.models import StructuredTransformerModel, TransformerConfig
    from structformer.models.bias import build_saab_bias, shuffle_field_ids_for_bias
    from structformer.tasks.msm import MSM_IGNORE_INDEX, mask_field_ids


def tiny_config(variant: str, *, head_type: str = "token", scale_embeddings: bool = False) -> "TransformerConfig":
    return TransformerConfig(
        vocab_size=32,
        field_vocab_size=6,
        max_length=8,
        variant=variant,
        head_type=head_type,
        num_labels=5 if head_type == "token" else 2,
        d_model=16,
        num_layers=2,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
        casa_rank=4,
        scale_embeddings=scale_embeddings,
    )


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class ModelForwardTests(unittest.TestCase):
    def setUp(self):
        self.input_ids = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]])
        self.field_ids = torch.tensor([[4, 4, 3, 0], [4, 3, 0, 0]])
        self.attention_mask = self.input_ids.ne(0)

    def test_baseline_token_forward_shape(self):
        model = StructuredTransformerModel(tiny_config("baseline"))
        model.eval()

        output = model(self.input_ids, self.field_ids, attention_mask=self.attention_mask)

        self.assertEqual(tuple(output.logits.shape), (2, 4, 5))
        self.assertEqual(tuple(output.hidden_states.shape), (2, 4, 16))
        self.assertIsNone(output.attentions)

    def test_saab_returns_structural_bias_when_requested(self):
        model = StructuredTransformerModel(tiny_config("saab"))
        model.eval()

        output = model(self.input_ids, self.field_ids, attention_mask=self.attention_mask, need_weights=True)

        self.assertEqual(len(output.attentions), 2)
        self.assertEqual(tuple(output.attentions[0].shape), (2, 4, 4, 4))
        self.assertEqual(len(output.structural_biases), 2)
        self.assertEqual(tuple(output.structural_biases[0].shape), (2, 4, 4))

    def test_baseline_and_saab_have_identical_seeded_initial_parameters(self):
        torch.manual_seed(1001)
        baseline = StructuredTransformerModel(tiny_config("baseline"))
        torch.manual_seed(1001)
        saab = StructuredTransformerModel(tiny_config("saab"))

        baseline_state = baseline.state_dict()
        saab_state = saab.state_dict()
        self.assertEqual(list(baseline_state), list(saab_state))
        for name in baseline_state:
            self.assertTrue(torch.equal(baseline_state[name], saab_state[name]), name)
        self.assertEqual(baseline.parameter_count(), saab.parameter_count())

    def test_masked_field_ids_feed_embeddings_and_saab_bias(self):
        original_field_ids = torch.tensor([[4, 4, 3, 3]])
        attention_mask = torch.ones_like(original_field_ids, dtype=torch.bool)
        masked_field_ids, labels, mask_positions = mask_field_ids(
            original_field_ids,
            attention_mask,
            mask_field_id=5,
            mask_probability=0.5,
            generator=torch.Generator().manual_seed(7),
        )
        model = StructuredTransformerModel(tiny_config("saab"))
        model.eval()
        embedded_field_ids = None

        def capture_embedding_input(_module, inputs):
            nonlocal embedded_field_ids
            embedded_field_ids = inputs[0].detach().clone()

        hook = model.embeddings.field_embeddings.register_forward_pre_hook(capture_embedding_input)
        try:
            with patch("structformer.models.model.build_saab_bias", wraps=build_saab_bias) as bias_builder:
                model(
                    torch.tensor([[2, 3, 4, 5]]),
                    masked_field_ids,
                    attention_mask=attention_mask,
                )
        finally:
            hook.remove()

        bias_field_ids = bias_builder.call_args.args[0]
        self.assertTrue(torch.equal(embedded_field_ids, masked_field_ids))
        self.assertTrue(torch.equal(bias_field_ids, masked_field_ids))
        self.assertTrue(torch.all(masked_field_ids[mask_positions].eq(5)))
        self.assertTrue(torch.equal(labels[mask_positions], original_field_ids[mask_positions]))
        self.assertTrue(torch.all(labels[~mask_positions].eq(MSM_IGNORE_INDEX)))

    def test_saab_shuffle_preserves_valid_pairs_and_padding(self):
        field_ids = torch.tensor(
            [
                [4, 4, 3, 3, 0, 0],
                [4, 3, 3, 0, 0, 0],
            ]
        )
        attention_mask = field_ids.ne(0)

        shuffled = shuffle_field_ids_for_bias(field_ids, attention_mask, seed=17)
        repeated = shuffle_field_ids_for_bias(field_ids, attention_mask, seed=17)

        self.assertTrue(torch.equal(shuffled, repeated))
        self.assertTrue(torch.equal(shuffled[~attention_mask], field_ids[~attention_mask]))
        for row in range(field_ids.shape[0]):
            valid = attention_mask[row]
            original_ids = field_ids[row, valid]
            shuffled_ids = shuffled[row, valid]
            self.assertEqual(
                sorted(original_ids.tolist()),
                sorted(shuffled_ids.tolist()),
            )
            original_pairs = original_ids.unsqueeze(0).eq(original_ids.unsqueeze(1)).sum()
            shuffled_pairs = shuffled_ids.unsqueeze(0).eq(shuffled_ids.unsqueeze(1)).sum()
            self.assertEqual(original_pairs.item(), shuffled_pairs.item())

    def test_saab_shuffle_is_independent_of_trailing_padding(self):
        short_ids = torch.tensor([[4, 4, 3, 3, 0, 0]])
        short_mask = short_ids.ne(0)
        long_ids = torch.tensor([[4, 4, 3, 3, 0, 0, 0, 0]])
        long_mask = long_ids.ne(0)

        short_shuffled = shuffle_field_ids_for_bias(short_ids, short_mask, seed=23)
        long_shuffled = shuffle_field_ids_for_bias(long_ids, long_mask, seed=23)

        self.assertTrue(torch.equal(short_shuffled[:, :4], long_shuffled[:, :4]))
        self.assertTrue(torch.equal(short_shuffled[:, 4:], short_ids[:, 4:]))
        self.assertTrue(torch.equal(long_shuffled[:, 4:], long_ids[:, 4:]))

    def test_model_passes_attention_mask_to_saab_shuffle(self):
        config = replace(tiny_config("saab"), saab_shuffle_bias=True)
        model = StructuredTransformerModel(config)
        model.eval()

        with patch(
            "structformer.models.model.shuffle_field_ids_for_bias",
            wraps=shuffle_field_ids_for_bias,
        ) as shuffle:
            model(
                self.input_ids,
                self.field_ids,
                attention_mask=self.attention_mask,
            )

        self.assertTrue(torch.equal(shuffle.call_args.args[0], self.field_ids))
        self.assertTrue(torch.equal(shuffle.call_args.args[1], self.attention_mask))

    def test_casa_forward_and_parameter_count(self):
        config = tiny_config("casa")
        model = StructuredTransformerModel(config)
        model.eval()

        output = model(self.input_ids, self.field_ids, attention_mask=self.attention_mask, need_weights=True)

        self.assertEqual(tuple(output.logits.shape), (2, 4, 5))
        self.assertEqual(len(output.structural_biases), 2)
        self.assertEqual(tuple(output.structural_biases[0].shape), (2, 4, 4))

        expected_per_layer = config.d_model * config.casa_rank + 2 * config.field_vocab_size * config.casa_rank + 1
        self.assertEqual(model.parameter_count()["casa"], expected_per_layer * config.num_layers)

    def test_sequence_head_forward_shape(self):
        model = StructuredTransformerModel(tiny_config("baseline", head_type="sequence"))
        model.eval()

        output = model(self.input_ids, self.field_ids, attention_mask=self.attention_mask)

        self.assertEqual(tuple(output.logits.shape), (2, 2))

    def test_embedding_scaling_is_opt_in(self):
        torch.manual_seed(123)
        unscaled = StructuredTransformerModel(tiny_config("baseline")).embeddings(self.input_ids, self.field_ids)
        torch.manual_seed(123)
        scaled = StructuredTransformerModel(tiny_config("baseline", scale_embeddings=True)).embeddings(
            self.input_ids,
            self.field_ids,
        )

        self.assertTrue(torch.allclose(scaled, unscaled * math.sqrt(16), atol=1.0e-6))


if __name__ == "__main__":
    unittest.main()
