# Paper Experiment Specification

This document defines the reported experiments generated from this reference
implementation. Configuration files are the executable source of truth.

## Controlled comparison

Baseline and SAAB share the same token embeddings, field embeddings,
positional embeddings, Transformer backbone, MSM prediction head, and
learnable parameter count. Their only intended difference is the attention
bias path:

- **Baseline:** field labels enter through field embeddings.
- **SAAB:** the same field labels also form a fixed attention-score bias of
  `+1.0` for same-field token pairs. The bias has no learnable parameters.

Unless an ablation explicitly says otherwise, SAAB applies this bias at every
layer with weight `1.0`.

## Masked Structure Modeling

MSM selects 15% of the positions in each represented field, replaces their
field IDs with the field-mask ID, and predicts the original field IDs. Token
IDs remain visible. The optimization objective is mean cross-entropy over the
masked positions.

## DBpedia data

- Source: the 560,000-example training split of DBpedia ontology
  classification.
- Fields: title and content.
- Split: a `numpy.random.RandomState(42)` permutation selects 490,000 training
  rows and 70,000 validation rows.
- The benchmark test split is not used.
- Maximum sequence length: 256.

`dbpedia/scripts/split_dbpedia.py` records source and output hashes in a split
manifest. `dbpedia/scripts/prepare_dbpedia.py` trains the tokenizer on the new
training split and writes the prepared JSONL files. The split script accepts
the original headerless CSV or an equivalent file with a
`label,title,content` header.

## PubMed data

- Source: PubMed 200k RCT.
- Fields: METHODS, RESULTS, and CONCLUSIONS.
- Only abstracts containing all three fields are retained.
- The fields are linearized in the order above.
- Maximum sequence length: 512.
- Reported seeds: 42, 123, and 1001.

## Shared paper-scale model and optimizer

- 4 Transformer layers unless depth is explicitly varied
- hidden size 768
- 6 attention heads
- feed-forward size 3072
- dropout 0.2
- scaled summed embeddings
- microbatch size 64
- 8 gradient-accumulation steps
- effective batch size 512
- AdamW, learning rate `1e-4`, weight decay `0.01`
- linear warmup for 50 steps followed by cosine decay
- minimum learning-rate ratio 0.1
- global gradient clipping at 1.0

DBpedia runs use 500 optimizer steps. PubMed runs use 1500 optimizer steps.
The depth experiment uses 750, 1000, and 1500 steps for 6, 8, and 12 layers,
respectively. Validation and checkpointing occur every 500 steps, with a final
validation pass also performed when a run ends between these intervals.

## Reported experiment matrix

- Eight-seed DBpedia comparison: 0, 7, 42, 99, 123, 256, 1001, 2024.
- Training trajectory: seed 1001 at steps 1, 50, 100, 200, 300, 500.
- MSM gradient comparison: seeds 42, 99, and 1001.
- Bias-strength sweep:
  - seed 1001 at weights 0.0, 0.25, 0.5, 1.0, 2.0;
  - seeds 42 and 123 at weights 0.0, 0.5, 1.0, 1.5, 2.0.
- Layer restriction, seed 1001:
  - all layers;
  - L3 only: `0,0,0,1`;
  - all except L3: `1,1,1,0`.
- Shuffled-bias control: seeds 99 and 1001.
- Depth: 6, 8, and 12 layers at seeds 99 and 1001.
- PubMed: seeds 42, 123, and 1001.

## Measurements

Same-field mass is the fraction of attention from a valid query token that is
directed to valid keys with the same original field label, averaged across
queries, heads, examples, and the selected evaluation batch. Attention entropy
and field-to-field mass are computed from the same saved attention
probabilities.

When enabled, layer-gradient logging records the joint L2 norm of the Q, K,
and V projection-weight gradients in every layer. These are gradients of the
MSM cross-entropy objective, measured after gradient accumulation and before
global gradient clipping. The reported L3/L2 ratio is the last-layer norm
divided by the penultimate-layer norm. The norms describe magnitude, not
gradient direction or the AdamW parameter-update magnitude.
