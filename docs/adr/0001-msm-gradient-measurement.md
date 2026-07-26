# ADR 0001: Measure Gradients from the MSM Objective

## Status

Accepted.

## Decision

Layer-gradient diagnostics use the gradients produced by the MSM
cross-entropy loss on the masked training batch. For each Transformer layer,
the diagnostic records the joint L2 norm over its Q, K, and V
projection-weight gradients after all microbatches have been accumulated and
before global gradient clipping.

The diagnostic is disabled by default and is enabled with
`--log-layer-gradients`.

## Reason

The diagnostic must measure the optimization signal that actually trains the
model. Backpropagating a synthetic scalar such as the sum of logits produces
a different signal and cannot support claims about MSM training dynamics.
Recording before clipping also preserves the relative magnitudes received by
the layers.

## Interpretation

The resulting norms measure magnitude only. They do not encode gradient
direction and are not the same as the parameter update produced by AdamW.
