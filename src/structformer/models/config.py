"""Configuration objects for model components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ModelVariant = Literal["baseline", "saab"]
HeadType = Literal["token", "sequence"]


@dataclass(frozen=True)
class TransformerConfig:
    """Minimal model config shared by Baseline and SAAB."""

    vocab_size: int
    field_vocab_size: int
    max_length: int
    variant: ModelVariant = "baseline"
    head_type: HeadType = "token"
    num_labels: int = 2

    d_model: int = 768
    num_layers: int = 4
    num_heads: int = 6
    ff_dim: int = 3072
    dropout: float = 0.2
    layer_norm_eps: float = 1e-5

    entity_vocab_size: int = 0
    value_type_vocab_size: int = 0
    time_vocab_size: int = 0
    pad_token_id: int = 0
    scale_embeddings: bool = False

    saab_field_weight: float = 1.0
    saab_entity_weight: float = 1.0
    saab_value_type_weight: float = 0.3
    saab_time_weight: float = 0.5

    # Per-layer bias mask for layer-restriction experiments.
    # Empty tuple (default) = uniform weight at every layer (saab_field_weight).
    # Non-empty = one multiplier per layer; length must equal num_layers.
    # Examples:
    #   (0.0, 0.0, 0.0, 1.0) → bias active only at L3
    #   (1.0, 1.0, 1.0, 0.0) → bias active at L0-L2, silent at L3
    saab_layer_mask: tuple[float, ...] = ()

    # Bias-content control: when True, the SAAB attention bias is built from a
    # per-example permutation of the field_ids supplied to the model. During MSM,
    # these are already the masked input IDs. The embeddings still receive the
    # unpermuted supplied field_ids, so no information or capacity is added or
    # removed; only the grouping the bias rewards is randomized.
    # Tests whether displacement needs a bias aligned with real structure or is
    # triggered by any fixed block-shaped attention prior of the same strength.
    saab_shuffle_bias: bool = False
    saab_shuffle_seed: int = 0

    def validate(self) -> None:
        if self.variant not in {"baseline", "saab"}:
            raise ValueError(f"Unknown model variant: {self.variant}")
        if self.head_type not in {"token", "sequence"}:
            raise ValueError(f"Unknown head type: {self.head_type}")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.field_vocab_size <= 0:
            raise ValueError("field_vocab_size must be positive")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.saab_layer_mask and len(self.saab_layer_mask) != self.num_layers:
            raise ValueError(
                f"saab_layer_mask has {len(self.saab_layer_mask)} entries "
                f"but model has {self.num_layers} layers; lengths must match"
            )
