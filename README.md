# How Structural Information Shapes Layer-wise Attention Routing in Transformers

A PyTorch reference implementation for comparing standard Transformer
attention with Structure-Aware Attention Bias (SAAB), with reproducible
experiments on DBpedia and PubMed 200k RCT.

## Overview

This repository studies whether changing how field information is made
available to attention alters layer-wise routing in a Transformer. It compares
two models with matched field-ID inputs, embeddings, Transformer backbone,
prediction head, trainable parameter count, and optimization protocol:

- **Baseline:** field IDs are supplied through learned field embeddings.
- **SAAB:** the same field IDs are embedded and additionally converted into a
  fixed pairwise same-field bias added to the attention logits.

Both models are trained with Masked Structure Modeling (MSM), a self-supervised
objective that masks selected field IDs while leaving token content visible and
asks the model to recover the original field labels. During MSM, both the field
embeddings and the SAAB bias receive the same post-masking field-ID tensor.

The implementation supports the reported DBpedia and PubMed experiments,
including same-field mass (SFM), attention entropy, field-to-field routing,
MSM loss-gradient measurements, paired uncertainty analyses, routing controls,
initialization sensitivity, and computational-overhead measurements.

## Scope of the release

This is a scripts-only research release. It includes the source code,
experiment configurations, preprocessing utilities, analysis scripts, tests,
and reproduction instructions required to regenerate the reported analyses
from locally supplied inputs.

Datasets, processed inputs, trained checkpoints, run directories, notebooks,
and generated result files are intentionally not distributed in this
repository.

## Documentation

- [Paper experiment specification](PAPER_EXPERIMENT_SPEC.md): model, data,
  training, evaluation, and analysis semantics.
- [Reproduction guide](REPRODUCE.md): complete commands for the reported
  experiments and analyses.
- [Reported environment](PAPER_ENVIRONMENT.md): software versions, hardware,
  and numerical settings recorded for the paper runs.

## Repository structure

```text
src/structformer/       model, task, data, training, logging, and metric code
dbpedia/configs/        paper-scale DBpedia configurations
dbpedia/scripts/        DBpedia splitting, preparation, and command generation
pubmed/configs/         paper-scale PubMed configurations
pubmed/scripts/         PubMed preparation
scripts/                diagnostics and statistical analysis
tests/                  correctness and release-integrity tests
```

## Installation

Python 3.10 or newer is required. Use an isolated environment and install the
PyTorch build appropriate for the target system before installing the remaining
dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

# Install the appropriate PyTorch build from:
# https://pytorch.org/get-started/locally/

pip install -e ".[runtime,dev]"
python scripts/check_env.py
```

To reconstruct the recorded non-PyTorch environment, install
`requirements-paper.txt` after manually installing the reported PyTorch/CUDA
build. Neither the project metadata nor its scripts install or replace PyTorch.

Training selects CUDA first and Apple MPS second. CPU training is rejected by
default. The `--allow-cpu` option is restricted to tests, diagnostic development,
and tiny smoke runs; final paper-scale runs are intended for CUDA.

## Data

The datasets are publicly available but are not downloaded automatically.
Core scripts accept local input and output paths.

### DBpedia

The reported training and validation inputs were generated from the
[`fancyzhx/dbpedia_14`](https://huggingface.co/datasets/fancyzhx/dbpedia_14)
mirror of the DBpedia ontology-classification dataset. The preparation scripts
expect CSV fields in `label,title,content` order and accept either a headerless
file or a file with that header.

The original 560,000-record training split is divided deterministically into
490,000 training records and 70,000 validation records using split seed 42.
The benchmark test split is excluded from training, validation, and checkpoint
selection. The untouched-test confirmation uses the original authors'
`dbpedia_csv/test.csv` distribution and the tokenizer retained from training.

### PubMed 200k RCT

Use the files from the
[PubMed 200k RCT repository](https://github.com/Franck-Dernoncourt/pubmed-rct).
Preparation retains abstracts containing METHODS, RESULTS, and CONCLUSIONS and
linearizes those fields in that order.

## Reproducing the experiments

Run commands from the repository root. Dataset preparation begins with:

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

The eight-seed DBpedia command generator emits the 16 matched Baseline/SAAB
training commands using seeds `0, 7, 42, 99, 123, 256, 1001, 2024`:

```bash
python dbpedia/scripts/multiseed_commands.py \
  --train-jsonl /path/to/processed/dbpedia/train.jsonl \
  --val-jsonl /path/to/processed/dbpedia/val.jsonl \
  --tokenizer-json /path/to/processed/dbpedia/tokenizer.json
```

Run each generated training command in a fresh process. Do not modify the shared
recipe between paired model variants. See [REPRODUCE.md](REPRODUCE.md) for the
PubMed runs, ablations, checkpoint diagnostics, paired statistical analyses,
untouched-test confirmation, initialization analysis, and overhead benchmark.

Each training run writes an inspectable local record containing its resolved
configuration, environment summary, data manifest, model summary, sample-batch
debug output, metrics, and checkpoints. Generated artifacts are excluded from
version control.

## Tests

```bash
PYTHONPATH=src python -m pytest -q
```

The tests cover model parity, masked-ID handling, SAAB bias construction,
deterministic data preparation, run logging, command generation, and the public
analysis interfaces.

## Citation

```bibtex
@unpublished{benamara2026saab,
  title  = {How Structural Information Shapes Layer-wise Attention Routing in Transformers},
  author = {Benamara, Amira and Sadeghzadeh, Arezoo and Kahraman, Fatih},
  year   = {2026},
  note   = {Manuscript under review}
}
```

Update the citation with the final journal metadata and DOI after publication.

## License

Released under the [MIT License](LICENSE).
