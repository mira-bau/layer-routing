# How Structural Information Shapes Layer-wise Attention Routing in Transformers

Reference implementation for the Baseline and Structure-Aware Attention Bias
(SAAB) experiments reported in the paper.

This repository compares two Transformers with the same visible field-ID inputs, backbone, prediction head, field embeddings, and
learnable parameter count. The Baseline receives field information through its
input embeddings. SAAB additionally constructs a fixed pairwise relation from
the visible post-masking field IDs and adds it to the attention logits. Both models
are trained with Masked Structure Modeling (MSM), which masks field labels and
asks the model to recover them.

The implementation includes the paper’s training variants and compact
diagnostics for same-field mass (SFM), attention entropy, field-to-field
attention, and MSM loss-gradient norms. See
[PAPER_EXPERIMENT_SPEC.md](PAPER_EXPERIMENT_SPEC.md) for the exact method and
[REPRODUCE.md](REPRODUCE.md) for the complete command sequence. The exact
recorded software and hardware environment is documented in
[PAPER_ENVIRONMENT.md](PAPER_ENVIRONMENT.md).

## Repository layout

```text
src/structformer/       shared model, task, data, training, and metric code
dbpedia/configs/        paper-scale DBpedia configurations
dbpedia/scripts/        deterministic split, preparation, and run generation
pubmed/configs/         paper-scale PubMed configurations
pubmed/scripts/         PubMed preparation
scripts/                environment, table, attention, and gradient analysis
tests/                  release and correctness checks
```

## Installation

Python 3.10 or newer is required. Use an isolated environment and install the
PyTorch build appropriate for the machine before installing the remaining
dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

# Install the appropriate PyTorch build from https://pytorch.org/get-started/locally/
pip install -e ".[runtime,dev]"
python scripts/check_env.py
```

For strict reconstruction of the recorded non-PyTorch package environment, use
`requirements-paper.txt` after manually installing the reported PyTorch/CUDA
build. The file does not install or replace PyTorch.

Training selects CUDA first and Apple MPS second. CPU training is rejected by
default; `--allow-cpu` is intended only for tests and tiny smoke runs. The
paper-scale runs were designed for CUDA.

## Datasets

The datasets are public but are not downloaded or redistributed by this
repository.

- **DBpedia ontology classification:** use the original benchmark files from
  [Crepe](https://github.com/zhangxiangxiao/Crepe), or the equivalent
  [DBpedia 14 dataset mirror](https://huggingface.co/datasets/fancyzhx/dbpedia_14).
  The expected CSV order is `label,title,content`; the original headerless CSV
  and an equivalent CSV with this header are both accepted.
- **PubMed 200k RCT:** use the files from the
  [official dataset repository](https://github.com/Franck-Dernoncourt/pubmed-rct).

The DBpedia experiments divide the original 560,000-example training split into
490,000 training examples and 70,000 validation examples with a deterministic
random split using seed 42. The benchmark test split is excluded from training, validation, and checkpoint selection; it is reserved for the documented untouched confirmatory analysis:

```bash
python dbpedia/scripts/split_dbpedia.py \
  --source-csv /path/to/dbpedia_csv/train.csv \
  --out-dir /path/to/dbpedia-paper-split

python dbpedia/scripts/prepare_dbpedia.py \
  --train-csv /path/to/dbpedia-paper-split/train.csv \
  --val-csv /path/to/dbpedia-paper-split/val.csv \
  --out-dir /path/to/processed/dbpedia
```

Prepare PubMed from its standard `LABEL<TAB>sentence` files:

```bash
python pubmed/scripts/prepare_pubmed.py \
  --train-txt /path/to/PubMed_200k_RCT/train.txt \
  --val-txt /path/to/PubMed_200k_RCT/dev.txt \
  --out-dir /path/to/processed/pubmed
```

The PubMed preparation retains abstracts containing METHODS, RESULTS, and
CONCLUSIONS, then linearizes those fields in that order. The reported PubMed
runs use 1500 optimization steps, with validation and checkpointing every 500
steps.

## Main DBpedia comparison

Use one shared recipe and override only the model variant, seed, data paths,
and output directory:

```bash
PYTHONPATH=src python -m structformer.training.train_msm \
  --config dbpedia/configs/msm_dbpedia_full_recipe.yaml \
  --model baseline \
  --seed 1001 \
  --train-jsonl /path/to/processed/dbpedia/train.jsonl \
  --val-jsonl /path/to/processed/dbpedia/val.jsonl \
  --tokenizer-json /path/to/processed/dbpedia/tokenizer.json \
  --run-dir runs/dbpedia_seed1001/baseline

PYTHONPATH=src python -m structformer.training.train_msm \
  --config dbpedia/configs/msm_dbpedia_full_recipe.yaml \
  --model saab \
  --seed 1001 \
  --train-jsonl /path/to/processed/dbpedia/train.jsonl \
  --val-jsonl /path/to/processed/dbpedia/val.jsonl \
  --tokenizer-json /path/to/processed/dbpedia/tokenizer.json \
  --run-dir runs/dbpedia_seed1001/saab
```

The eight-seed command generator defaults to the paper’s seeds
`0, 7, 42, 99, 123, 256, 1001, 2024`:

```bash
python dbpedia/scripts/multiseed_commands.py \
  --train-jsonl /path/to/processed/dbpedia/train.jsonl \
  --val-jsonl /path/to/processed/dbpedia/val.jsonl \
  --tokenizer-json /path/to/processed/dbpedia/tokenizer.json
```

## MSM gradient measurement

Enable `--log-layer-gradients` during a run to record the joint Q/K/V
projection-weight gradient norm for each layer. The value is taken from the
MSM cross-entropy loss after gradient accumulation and before global gradient
clipping:

```bash
PYTHONPATH=src python -m structformer.training.train_msm \
  --config dbpedia/configs/msm_dbpedia_full_recipe.yaml \
  --model saab \
  --seed 1001 \
  --train-jsonl /path/to/processed/dbpedia/train.jsonl \
  --val-jsonl /path/to/processed/dbpedia/val.jsonl \
  --tokenizer-json /path/to/processed/dbpedia/tokenizer.json \
  --log-layer-gradients \
  --run-dir runs/gradients/saab_seed1001

PYTHONPATH=src python scripts/analyze_training_gradients.py \
  --run-root runs/gradients \
  --seed 1001 \
  --out-dir outputs/gradients/seed1001
```

The analyzer rejects logs produced from another objective or measurement
point. Gradient norms measure magnitude, not gradient direction or the AdamW
parameter update.

## Tests

```bash
PYTHONPATH=src pytest -q
```

## Citation

```bibtex
@unpublished{benamara2026saab,
  title  = {How Structural Information Shapes Layer-wise Attention Routing in Transformers},
  author = {Benamara, Amira and Sadeghzadeh, Arezoo and Kahraman, Fatih},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
