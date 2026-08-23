# Paper Experiment Specification

This document defines the reported experiments generated from this reference
implementation. Configuration files are the executable source of truth.

## Controlled comparison

Baseline and SAAB share the same token embeddings, field embeddings,
positional embeddings, Transformer backbone, MSM prediction head, and
learnable parameter count. They share visible field-ID inputs, learned components, and the training
protocol; SAAB additionally changes the attention-bias path:

- **Baseline:** field labels enter through field embeddings.
- **SAAB:** the same field labels also form a fixed attention-score bias of
  `+1.0` for same-field token pairs. The bias has no learnable parameters.

Unless an ablation explicitly says otherwise, SAAB applies this bias at every
layer with weight `1.0`.

## Masked Structure Modeling

For each micro-batch and each field ID represented among non-padding positions,
MSM samples without replacement min(n, max(1, round(0.15n))) positions, replaces
their field IDs with one shared mask-field ID, and predicts the original IDs.
The masked IDs are supplied to both field embeddings and the SAAB bias; original
IDs are targets only. Token IDs remain visible, and the objective is mean
cross-entropy over selected positions.

## DBpedia data

- Source: the 560,000-example training split of DBpedia ontology
  classification.
- Fields: title and content.
- Split: a `numpy.random.RandomState(42)` permutation selects 490,000 training
  rows and 70,000 validation rows.
- The benchmark test split is excluded from training, validation, and checkpoint
selection; the retained seed-1001 checkpoints support the documented untouched-test confirmation.
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

The primary PubMed routing comparison evaluates the first 64 validation
records with a 256-token evaluation cap. The configured-length sensitivity
analysis reevaluates the same examples at the model maximum of 512 tokens.

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
- Training trajectory: seed 1001 at steps 1, 10, 50, 100, 200, 300, 500.
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
- Opportunity-adjusted SFM: the fixed seed-1001 DBpedia and PubMed pairs,
  including PubMed at both 256 and 512 evaluation tokens.
- Untouched DBpedia test confirmation: the retained seed-1001 checkpoint pair
  on a deterministic reservoir sample of 2,336 records from the original
  70,000-record benchmark test split, selected with analysis seed 1001.
- Individual-token attention: one output-independently selected record from
  that same untouched-test reservoir, restricted to 20--30 valid tokens with
  at least four tokens from each named field; layers L2 and L3 are reported.
- Exploratory initialization sensitivity: the same eight seeds and final
  routing outcomes used by the multi-seed comparison. Five simple initial-state
  summaries are examined: title--content field-embedding distance,
  mask-to-named-field-centroid distance, and the L3/L2 ratios of joint Q/K/V,
  attention-output, and feed-forward weight norms. A one-update MSM probe also
  records the SAAB-minus-Baseline changes in the L3/L2 joint-Q/K/V gradient and
  AdamW-update ratios.
- Robust initial-loss sensitivity: 256 fixed validation records divided into
  four non-overlapping 64-record blocks, with dropout disabled and deterministic
  MSM mask seeds 101, 202, 303, 404, and 505. The analysis reports per-mask and
  per-block results, exclusion of the first 64 records, and leave-one-seed-out
  descriptive correlations with final displacement.
- Computational overhead: freshly initialized matched models at sequence
  lengths 64, 128, and 256 with batch size 4 on an NVIDIA A100-SXM4-40GB GPU.
  Measurements use float32 with automatic mixed precision and TF32 disabled,
  five timed blocks, 20 inference iterations per block, and five complete
  training-step iterations per block.

## Measurements

Same-field mass is the fraction of attention from a valid query token that is
directed to valid keys with the same original field label, averaged across
queries, heads, examples, and the selected evaluation batch. Attention entropy
and field-to-field mass are computed from the same saved attention
probabilities.

The paired fixed-model analyses use the first 64 held-out validation examples,
10,000 paired-bootstrap resamples, 20,000 paired sign-flip permutations, Holm
correction across the four layers, and analysis seed 1001. These intervals and
tests quantify variation across examples conditional on one trained model pair;
they do not quantify variation across independently trained models.

The length analysis separately selects 2,336 validation records per dataset by
deterministic reservoir sampling. DBpedia uses its configured 256-token length;
PubMed length associations use the configured 512-token evaluation. Exact-length
standardization uses only token lengths observed in both datasets.

When enabled, layer-gradient logging records the joint L2 norm of the Q, K,
and V projection-weight gradients in every layer. These are gradients of the
MSM cross-entropy objective, measured after gradient accumulation and before
global gradient clipping. The reported L3/L2 ratio is the last-layer norm
divided by the penultimate-layer norm. The norms describe magnitude, not
gradient direction or the AdamW parameter-update magnitude.

The individual-token diagnostic is qualitative. Record selection is independent
of attention outputs, attention is averaged over all heads, and the result is
not used as inferential evidence. The initialization analysis is likewise
exploratory and descriptive over eight already-observed final outcomes; it is
not a causal analysis or a validated predictor for new initializations.
