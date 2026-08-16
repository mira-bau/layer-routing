"""Top-level structured Transformer model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from structformer.models.bias import (
    SAABWeights,
    build_saab_bias,
    shuffle_field_ids_for_bias,
)
from structformer.models.config import TransformerConfig
from structformer.models.embeddings import StructuredEmbeddings
from structformer.models.encoder import TransformerEncoder
from structformer.models.heads import SequenceClassificationHead, TokenClassificationHead


@dataclass
class ModelOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    attentions: list[torch.Tensor] | None = None
    structural_biases: list[torch.Tensor] | None = None


class StructuredTransformerModel(nn.Module):
    """Shared backbone plus task head for baseline, SAAB, or CASA."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embeddings = StructuredEmbeddings(config)
        self.encoder = TransformerEncoder(config)

        if config.head_type == "token":
            self.head = TokenClassificationHead(config.d_model, config.num_labels)
        else:
            self.head = SequenceClassificationHead(config.d_model, config.num_labels, config.dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        field_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        entity_ids: torch.Tensor | None = None,
        value_type_ids: torch.Tensor | None = None,
        time_ids: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> ModelOutput:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)

        hidden_states = self.embeddings(
            input_ids,
            field_ids,
            entity_ids=entity_ids,
            value_type_ids=value_type_ids,
            time_ids=time_ids,
        )

        fixed_bias: torch.Tensor | list[torch.Tensor | None] | None = None
        if self.config.variant == "saab":
            # Bias-content control: build the bias from a grouping shuffled only
            # across attended positions. Embeddings keep the unshuffled input
            # field_ids, including the MSM mask-field ID at selected positions.
            bias_field_ids = field_ids
            if self.config.saab_shuffle_bias:
                bias_field_ids = shuffle_field_ids_for_bias(
                    field_ids,
                    attention_mask,
                    self.config.saab_shuffle_seed,
                )
            base_bias = build_saab_bias(
                bias_field_ids,
                entity_ids=entity_ids,
                value_type_ids=value_type_ids,
                time_ids=time_ids,
                weights=SAABWeights(
                    field=self.config.saab_field_weight,
                    entity=self.config.saab_entity_weight,
                    value_type=self.config.saab_value_type_weight,
                    time=self.config.saab_time_weight,
                ),
            )
            if self.config.saab_layer_mask:
                # Build one bias tensor per layer; None where the mask is zero
                # so the attention module skips the addition entirely.
                fixed_bias = [
                    base_bias * m if m != 0.0 else None
                    for m in self.config.saab_layer_mask
                ]
            else:
                fixed_bias = base_bias

        encoder_output = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            field_ids=field_ids,
            fixed_structural_bias=fixed_bias,
            need_weights=need_weights,
        )
        logits = self.head(encoder_output.hidden_states, attention_mask)
        return ModelOutput(
            logits=logits,
            hidden_states=encoder_output.hidden_states,
            attentions=encoder_output.attentions,
            structural_biases=encoder_output.structural_biases,
        )

    def parameter_count(self) -> dict[str, int]:
        """Return parameter counts useful for run summaries."""

        counts = {
            "total": sum(param.numel() for param in self.parameters()),
            "trainable": sum(param.numel() for param in self.parameters() if param.requires_grad),
        }
        casa = sum(param.numel() for name, param in self.named_parameters() if ".casa." in name)
        if casa:
            counts["casa"] = casa
        return counts
