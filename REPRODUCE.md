# Reproducing the Paper Experiments

Run all commands from the repository root with `PYTHONPATH=src`. Paths below
are placeholders for locally prepared artifacts; the scripts do not download
data.

Before starting, complete the installation and environment check in
[README.md](README.md#installation). Paper-scale commands require locally
supplied datasets and checkpoints; the repository intentionally contains
neither. Keep Baseline and SAAB runs for a seed on the same device and do not
change the shared recipe between the two variants.

## Workflow and outputs at a glance

| Manuscript analysis | Command or section | Primary generated output |
| --- | --- | --- |
| Environment and implementation check | Smoke test below | `runs/smoke/final_summary.json` |
| Dataset length distribution | Section 1 | `outputs/dataset_lengths/length_summary.csv` |
| Eight-seed DBpedia performance | Section 2 | `outputs/tables/dbpedia_multiseed/final_metrics.csv` |
| Routing trajectory | Section 3 | `outputs/timeseries/seed*/timeseries_diagnostics.json` |
| MSM gradient analysis | Section 3 | `outputs/gradients/seed*/gradient_checkpoint_summary.csv` |
| Bias, placement, alignment, and depth controls | Sections 4–5 | One `attention_diagnostics.json` per completed pair |
| PubMed comparison | Sections 6 and 8 | `outputs/paired_sfm/paired_primary_stats.csv` |
| Eight-seed routing profiles | Section 10, first two commands | `outputs/dbpedia_final_sfm/final_sfm_by_seed.csv` |
| Paired uncertainty and length analyses | Section 8 | `outputs/paired_sfm/analysis_summary.json` |
| Opportunity and exact-length adjustments | Section 8 | `opportunity_adjusted_primary_stats.csv` and `length_standardization.json` |
| Untouched DBpedia test and token-level figure | Section 9 | `analysis.json` and `individual_token_attention_candidate.pdf` |
| Initialization sensitivity | Section 10 | `initialization_correlations.csv` |
| Computational overhead | Section 11 | `baseline_saab_overhead.csv` |

Resource categories:

- `scripts/check_env.py`, tests, and the smoke run are short software checks.
- Dataset preparation is CPU preprocessing.
- Paper-scale training and checkpoint attention analysis target CUDA; the
  reported environment used one NVIDIA A100-SXM4-40GB.
- Statistical resampling scripts are post-processing, but their configured
  bootstrap and permutation counts are intentionally substantial.

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

The examples use this local source layout:

```text
local-data/dbpedia_csv/train.csv
local-data/dbpedia_csv/test.csv
local-data/PubMed_200k_RCT/train.txt
local-data/PubMed_200k_RCT/dev.txt
```

The DBpedia CSVs use `label,title,content` columns, with or without a header.
If using the PubMed source repository, extract
`PubMed_200k_RCT/train.7z` before continuing.

```bash
7z x local-data/PubMed_200k_RCT/train.7z \
  -olocal-data/PubMed_200k_RCT
```

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

Successful preparation prints `Prepared ... MSM artifacts` and produces:

```text
/path/to/processed/{dbpedia,pubmed}/
├── train.jsonl
├── val.jsonl
├── tokenizer.json
├── field_vocab.json
└── manifest.json
```

Before training, inspect both `manifest.json` files and confirm that the
DBpedia split contains 490,000 training and 70,000 validation records, while
the retained PubMed split contains 177,533 training and 2,336 validation
records.

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

After each Baseline/control or Baseline/SAAB pair finishes, evaluate the fixed
32-example routing batch with `scripts/diag_attention.py`. Substitute the pair's
actual checkpoint and output paths:

```bash
PYTHONPATH=src python scripts/diag_attention.py \
  --baseline-ckpt /path/to/baseline/checkpoints/latest.pt \
  --saab-ckpt /path/to/comparison/checkpoints/latest.pt \
  --val-jsonl /path/to/processed/dbpedia/val.jsonl \
  --field-vocab-json /path/to/processed/dbpedia/field_vocab.json \
  --n-examples 32 \
  --device cuda \
  --out-dir outputs/diagnostics/experiment_name
```

The table-ready layer values are in
`outputs/diagnostics/experiment_name/attention_diagnostics.json`.

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

A completed paper-scale run directory has at least:

```text
run-directory/
├── resolved_config.json
├── environment.json
├── data_manifest.json
├── model_summary.json
├── sample_batch.json
├── sample_rows.json
├── metrics.jsonl
├── metrics.csv
├── final_summary.json
└── checkpoints/
    └── latest.pt
```

Runs requested with `--diagnostic-steps` also contain `step_XXXX.pt` files.
Confirm `final_summary.json` reports the requested final step before using a
checkpoint in an analysis.

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

First compute the final, unrounded, 32-example attention diagnostics for all
eight DBpedia seed pairs:

```bash
for seed in 0 7 42 99 123 256 1001 2024; do
  PYTHONPATH=src python scripts/diag_attention.py \
    --baseline-ckpt "runs/dbpedia_multiseed/baseline_seed${seed}/checkpoints/latest.pt" \
    --saab-ckpt "runs/dbpedia_multiseed/saab_seed${seed}/checkpoints/latest.pt" \
    --val-jsonl /path/to/processed/dbpedia/val.jsonl \
    --field-vocab-json /path/to/processed/dbpedia/field_vocab.json \
    --n-examples 32 \
    --device cuda \
    --out-dir "outputs/dbpedia_final_sfm/seed${seed}"
done
```

Build the initialization-analysis input directly from those JSON files. The
extractor verifies every seed, the four-layer shape, and checkpoint step 500:

```bash
PYTHONPATH=src python scripts/extract_final_sfm_outcomes.py \
  --diagnostics-root outputs/dbpedia_final_sfm \
  --seeds 0,7,42,99,123,256,1001,2024 \
  --expected-step 500 \
  --output outputs/dbpedia_final_sfm/final_sfm_by_seed.csv
```

Then run the bounded initialization analysis:

```bash
PYTHONPATH=src python scripts/analyze_initialization_sensitivity.py \
  --config dbpedia/configs/msm_dbpedia_full_recipe.yaml \
  --validation-jsonl /path/to/processed/dbpedia/val.jsonl \
  --final-outcomes-csv outputs/dbpedia_final_sfm/final_sfm_by_seed.csv \
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
validated predictor for new initializations. A successful run prints
`initialization-sensitivity complete` and writes its main reported association
to `outputs/initialization_sensitivity/initialization_correlations.csv`.

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

## Completion checklist

A reproduction is internally complete when:

- the environment check and test suite pass;
- the dataset manifests contain the expected record counts;
- each required Baseline/SAAB pair reached the same requested final step;
- paired runs report identical trainable parameter counts and the intended
  seed, data sample, masking, and optimizer settings;
- every analysis command ends with its documented completion message and its
  primary output from the workflow table exists; and
- generated datasets, checkpoints, and results remain outside version control.
