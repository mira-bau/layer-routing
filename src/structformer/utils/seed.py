"""Reproducibility helpers."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when those libraries are available."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        np = None
    if np is not None:
        np.random.seed(seed)

    try:
        import torch  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

