# Reproducing the Paper Experiments

Run all commands from the repository root with `PYTHONPATH=src`. Paths below
are placeholders for locally prepared artifacts; the scripts do not download
data.

## Quick end-to-end smoke test

This tiny synthetic run checks model construction, MSM masking, optimization,
and artifact logging without downloading data. It is not a paper experiment:

```bash
PYTHONPATH=src python -m structformer.training.smoke_msm \
  --config dbpedia/configs/smoke_msm.yaml \
  --max-steps 2 \
  --allow-cpu \
  --run-dir runs/smoke
```

## 1. Prepare the datasets

```bash
python dbpedia/scripts/split_dbpedia.py \
  --source-csv /path/to/dbpedia_csv/train.csv \
  --out-dir /path/to/dbpedia-paper-split

python dbpedia/scripts/prepare_dbpedia.py \
  --train-csv /path/to/dbpedia-paper-split/train.csv \
  --val-csv /path/to/dbpedia-paper-split/val.csv \
  --out-dir /path/to/processed/dbpedia

python pubmed/scripts/prepare_pubmed.py \
  --train-txt /path/to/PubMed_200k_RCT/train.txt \
  --val-txt /path/to/PubMed_200k_RCT/dev.txt \
  --out-dir /path/to/processed/pubmed
```

Keep the generated manifests with the run artifacts.

Summarize the post-tokenization length distributions reported for the prepared
training and validation splits:

```bash
PYTHONPATH=src python scripts/analyze_dataset_lengths.py \
  --dbpedia-prepared-dir /path/to/processed/dbpedia \
  --pubmed-prepared-dir /path/to/processed/pubmed \
  --splits train,val \
  --out-dir outputs/dataset_lengths
```

## 2. Run the eight-seed DBpedia comparison

The generator prints the 16 Baseline/SAAB commands and the table command:

```bash
python dbpedia/scripts/multiseed_commands.py \
  --train-jsonl /path/to/processed/dbpedia/train.jsonl \
  --val-jsonl /path/to/processed/dbpedia/val.jsonl \
  --tokenizer-json /path/to/processed/dbpedia/tokenizer.json
```

Run the printed training commands in fresh processes. Do not edit the shared
recipe between variants.

## 3. Record the training trajectory and MSM gradients

For each model and seed in `42, 99, 1001`, run the DBpedia recipe with both
diagnostic checkpointing and gradient logging:

```bash
PYTHONPATH=src python -m structformer.training.train_msm \
  --config dbpedia/configs/msm_dbpedia_full_recipe.yaml \
  --model baseline \
  --seed 1001 \
  --train-jsonl /path/to/processed/dbpedia/train.jsonl \
  --val-jsonl /path/to/processed/dbpedia/val.jsonl \
  --tokenizer-json /path/to/processed/dbpedia/tokenizer.json \
  --diagnostic-steps 1,10,50,100,200,300,500 \
  --log-layer-gradients \
  --run-dir runs/gradients/baseline_seed1001
```

Repeat with `--model saab` and `--run-dir
runs/gradients/saab_seed1001`, then repeat the pair for the other two seeds.
Analyze each pair:

```bash
PYTHONPATH=src python scripts/analyze_training_gradients.py \
  --run-root runs/gradients \
  --seed 1001 \
  --out-dir outputs/gradients/seed1001

PYTHONPATH=src python scripts/diag_timeseries.py \
  --run-root runs/gradients \
  --seed 1001 \
  --val-jsonl /path/to/processed/dbpedia/val.jsonl \
  --field-vocab-json /path/to/processed/dbpedia/field_vocab.json \
  --n-examples 32 \
  --device cuda \
  --out-dir outputs/timeseries/seed1001
```

## 4. Run the SAAB ablations

All commands below use
`dbpedia/configs/msm_dbpedia_full_recipe.yaml`, the same prepared DBpedia
paths, `--model saab`, and a distinct `--run-dir`.

Bias-strength options:

```text
seed 1001: --saab-field-weight 0.0, 0.25, 0.5, 1.0, or 2.0
seed 42:   --saab-field-weight 0.0, 0.5, 1.0, 1.5, or 2.0
seed 123:  --saab-field-weight 0.0, 0.5, 1.0, 1.5, or 2.0
```

Layer-restriction options at seed 1001:

```text
SAAB-all:    --saab-layer-mask 1,1,1,1
SAAB-L3only: --saab-layer-mask 0,0,0,1
SAAB-notL3:  --saab-layer-mask 1,1,1,0
```

Shuffled-bias control at seeds 99 and 1001:

```text
--saab-shuffle-bias --saab-shuffle-seed 0
```

For this control, MSM masking occurs first. The bias-construction IDs are then
permuted deterministically only among valid non-padding positions in each
example. This preserves the valid-position ID multiset and the number and
magnitude of positive valid-token bias pairs, while the field embeddings retain
the unpermuted masked IDs.

For example:

```bash
PYTHONPATH=src python -m structformer.training.train_msm \
  --config dbpedia/configs/msm_dbpedia_full_recipe.yaml \
  --model saab \
  --seed 1001 \
  --train-jsonl /path/to/processed/dbpedia/train.jsonl \
  --val-jsonl /path/to/processed/dbpedia/val.jsonl \
  --tokenizer-json /path/to/processed/dbpedia/tokenizer.json \
  --saab-layer-mask 1,1,1,0 \
  --run-dir runs/restriction/saab_notL3_seed1001
```

## 5. Run the depth experiment

Use seeds 99 and 1001. Override the layer count, optimizer-step count, and
diagnostic checkpoints as follows for both Baseline and SAAB:

```text
6 layers:
  --num-layers 6
  --max-steps 750
  --diagnostic-steps 1,50,100,200,300,500,750

8 layers:
  --num-layers 8
  --max-steps 1000
  --diagnostic-steps 1,50,100,200,300,500,750,1000

12 layers:
  --num-layers 12
  --max-steps 1500
  --diagnostic-steps 1,50,100,200,300,500,750,1000,1250,1500
```

The final diagnostic checkpoint is the one used for the reported routing
analysis. All other settings come from the DBpedia paper recipe.

## 6. Run PubMed

Run both models at seeds 42, 123, and 1001:

```bash
PYTHONPATH=src python -m structformer.training.train_msm \
  --config pubmed/configs/msm_pubmed_full_recipe.yaml \
  --model saab \
  --seed 1001 \
  --train-jsonl /path/to/processed/pubmed/train.jsonl \
  --val-jsonl /path/to/processed/pubmed/val.jsonl \
  --tokenizer-json /path/to/processed/pubmed/tokenizer.json \
  --run-dir runs/pubmed/saab_seed1001
```

Repeat with `--model baseline` and the other seeds.

## 7. Inspect the artifacts

Each run writes the resolved configuration, environment summary, data
manifest, model and parameter summary, sample-batch debug output, JSONL/CSV
metrics, and checkpoints. Preserve these files when archiving a reported run.

Use `scripts/diag_attention.py` for per-layer SFM, entropy, and field-to-field
attention on paired final checkpoints. Use `scripts/make_tables.py` to collect
training and validation metrics from compatible run directories.

## 8. Paired SFM, uncertainty, and length analysis

Use the fixed seed-1001 final checkpoint pairs. This command evaluates the
first 64 validation examples for the primary paired comparisons and a
deterministic 2,336-record sample for the length analyses. It evaluates PubMed
at both the primary 256-token cap and the configured 512-token maximum.

```bash
PYTHONPATH=src python scripts/analyze_paired_sfm_length.py \
  --dbpedia-baseline-checkpoint runs/dbpedia_multiseed/baseline_seed1001/checkpoints/latest.pt \
  --dbpedia-saab-checkpoint runs/dbpedia_multiseed/saab_seed1001/checkpoints/latest.pt \
  --dbpedia-validation-jsonl /path/to/processed/dbpedia/val.jsonl \
  --dbpedia-field-vocab-json /path/to/processed/dbpedia/field_vocab.json \
  --pubmed-baseline-checkpoint runs/pubmed/baseline_seed1001/checkpoints/latest.pt \
  --pubmed-saab-checkpoint runs/pubmed/saab_seed1001/checkpoints/latest.pt \
  --pubmed-validation-jsonl /path/to/processed/pubmed/val.jsonl \
  --pubmed-field-vocab-json /path/to/processed/pubmed/field_vocab.json \
  --primary-examples 64 \
  --length-sample-size 2336 \
  --bootstrap-resamples 10000 \
  --correlation-bootstrap-resamples 2000 \
  --permutations 20000 \
  --analysis-seed 1001 \
  --device cuda \
  --out-dir outputs/paired_sfm
```

Apply the field-opportunity adjustment to those retained per-example values:

```bash
PYTHONPATH=src python scripts/analyze_opportunity_normalized_sfm.py \
  --analysis-dir outputs/paired_sfm \
  --dbpedia-validation-jsonl /path/to/processed/dbpedia/val.jsonl \
  --pubmed-validation-jsonl /path/to/processed/pubmed/val.jsonl \
  --bootstrap-resamples 10000 \
  --permutations 20000 \
  --analysis-seed 1001 \
  --out-dir outputs/opportunity_adjusted_sfm
```

Standardize both datasets to their exactly shared token lengths:

```bash
PYTHONPATH=src python scripts/analyze_length_standardized_sfm.py \
  --dbpedia-csv outputs/paired_sfm/dbpedia/per_example_sfm.csv \
  --pubmed-csv outputs/paired_sfm/pubmed/configured_length_per_example_sfm.csv \
  --band-widths 1,2,5,10 \
  --bootstrap-resamples 10000 \
  --seed 1001 \
  --output outputs/length_standardization.json
```

## 9. Untouched DBpedia test confirmation

Encode the original 70,000-record benchmark test CSV with the retained training
tokenizer. This does not retrain or modify the tokenizer.

```bash
PYTHONPATH=src python dbpedia/scripts/prepare_external_dbpedia_split.py \
  --input-csv /path/to/dbpedia_csv/test.csv \
  --tokenizer-json /path/to/processed/dbpedia/tokenizer.json \
  --output-jsonl /path/to/processed/dbpedia/test.jsonl \
  --manifest /path/to/processed/dbpedia/test_manifest.json \
  --split-name test \
  --max-length 256 \
  --expected-records 70000
```

Run the predeclared 2,336-record reservoir confirmation with the retained
seed-1001 checkpoints:

```bash
PYTHONPATH=src python scripts/analyze_untouched_dbpedia_test.py \
  --baseline-checkpoint runs/dbpedia_multiseed/baseline_seed1001/checkpoints/latest.pt \
  --saab-checkpoint runs/dbpedia_multiseed/saab_seed1001/checkpoints/latest.pt \
  --test-jsonl /path/to/processed/dbpedia/test.jsonl \
  --test-manifest /path/to/processed/dbpedia/test_manifest.json \
  --field-vocab-json /path/to/processed/dbpedia/field_vocab.json \
  --sample-size 2336 \
  --bootstrap-resamples 10000 \
  --permutations 20000 \
  --analysis-seed 1001 \
  --out-dir outputs/untouched_dbpedia_test
```

The qualitative individual-token figure uses an output-independent record
selection from the same reservoir:

```bash
PYTHONPATH=src python scripts/analyze_individual_token_attention.py \
  --baseline-checkpoint runs/dbpedia_multiseed/baseline_seed1001/checkpoints/latest.pt \
  --saab-checkpoint runs/dbpedia_multiseed/saab_seed1001/checkpoints/latest.pt \
  --test-jsonl /path/to/processed/dbpedia/test.jsonl \
  --test-manifest /path/to/processed/dbpedia/test_manifest.json \
  --field-vocab-json /path/to/processed/dbpedia/field_vocab.json \
  --sample-size 2336 \
  --analysis-seed 1001 \
  --min-length 20 \
  --max-length 30 \
  --min-tokens-per-field 4 \
  --layers 2,3 \
  --top-changes 20 \
  --device cuda \
  --out-dir outputs/individual_token_attention
```

## 10. Exploratory initialization sensitivity

Prepare a local CSV containing unrounded final SFM outcomes for the eight seed
pairs. It may contain either `seed,delta_l2,delta_l3` or
`seed,baseline_l2,baseline_l3,saab_l2,saab_l3`; an optional `pattern` column is
retained for inspection. Then run:

```bash
PYTHONPATH=src python scripts/analyze_initialization_sensitivity.py \
  --config dbpedia/configs/msm_dbpedia_full_recipe.yaml \
  --validation-jsonl /path/to/processed/dbpedia/val.jsonl \
  --final-outcomes-csv /path/to/local/final_sfm_by_seed.csv \
  --seeds 0,7,42,99,123,256,1001,2024 \
  --mask-seeds 101,202,303,404,505 \
  --probe-examples 256 \
  --probe-blocks 4 \
  --vocab-size 30000 \
  --device cpu \
  --allow-cpu \
  --out-dir outputs/initialization_sensitivity
```

This analysis is descriptive over the same eight final outcomes and is not a
validated predictor for new initializations.

## 11. Computational-overhead benchmark

Run this command on an NVIDIA A100-SXM4-40GB GPU. Omitting `--allow-tf32`
keeps TF32 disabled; the script uses float32 without automatic mixed precision.

```bash
PYTHONPATH=src python scripts/benchmark_computational_overhead.py \
  --device cuda \
  --lengths 64,128,256 \
  --batch-size 4 \
  --inference-warmup 5 \
  --inference-iterations 20 \
  --training-warmup 2 \
  --training-iterations 5 \
  --repeats 5 \
  --seed 2026 \
  --vocab-size 30000 \
  --field-vocab-size 6 \
  --num-labels 5 \
  --d-model 768 \
  --num-layers 4 \
  --num-heads 6 \
  --ff-dim 3072 \
  --dropout 0.2 \
  --out-dir outputs/computational_overhead
```

All analysis outputs are local generated artifacts and remain excluded from the
repository. Preserve their generated manifests and hash records with the local
run archive.
