# Reproducing the Paper Experiments

Run all commands from the repository root with `PYTHONPATH=src`. Paths below
are placeholders for locally prepared artifacts; the scripts do not download
data.

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

## 8. Reviewer-facing analyses

The following scripts operate on retained run artifacts and local prepared data;
they do not download datasets. Use the paired final checkpoints and record the
resulting manifests with the outputs.

```bash
# Paired SFM statistics, bootstrap CIs, permutation tests, and length associations
PYTHONPATH=src python scripts/analyze_paired_sfm_length.py --help

# Opportunity-normalized SFM
PYTHONPATH=src python scripts/analyze_opportunity_normalized_sfm.py --help

# Exact-length standardization across DBpedia and PubMed
PYTHONPATH=src python scripts/analyze_length_standardized_sfm.py --help

# Untouched DBpedia test confirmation using retained fixed checkpoints
PYTHONPATH=src python scripts/analyze_untouched_dbpedia_test.py --help

# Computational-overhead benchmark
PYTHONPATH=src python scripts/benchmark_computational_overhead.py --help
```

Each command exposes explicit local input and output paths. Inspect `--help`
before launching a paper-scale run, and preserve its generated configuration,
manifest, metrics, and hash records with the analysis output.
