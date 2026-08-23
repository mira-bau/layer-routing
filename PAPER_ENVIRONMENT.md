# Reported Paper Environment

The reported training and evaluation runs recorded the following environment:

- Python 3.12.13
- PyTorch 2.11.0 with CUDA 12.8 (`2.11.0+cu128`)
- NumPy 2.0.2
- pandas 2.2.2
- PyYAML 6.0.3
- scikit-learn 1.6.1
- tokenizers 0.22.2
- NVIDIA A100-SXM4-40GB
- Linux with glibc 2.35

Training and evaluation used float32 without automatic mixed precision or
gradient scaling. No explicit TF32 override was applied to the training runs;
both variants within a seed pair used the same numerical configuration. The
computational-overhead benchmark separately disabled both cuDNN and matrix
multiplication TF32.

Install the PyTorch build manually using the instructions appropriate for the
target CUDA runtime. The repository does not install or replace PyTorch. After
PyTorch is available, the recorded non-PyTorch packages can be installed with:

```bash
pip install -r requirements-paper.txt
```

The training environment report did not record the Matplotlib version.
Matplotlib was used for generated figures, not for model training or numerical
metric calculation, so `requirements-paper.txt` lists it without inventing an
unverified version pin. Every run writes its observed environment to the local
run directory; retain that file with reproduced outputs.

Exact package equality does not guarantee bitwise-identical CUDA execution.
The reported comparisons pair Baseline and SAAB within the same environment,
initialization, data order, MSM masking stream, and optimization protocol.
