"""Synthetic classification data with a controllable signal. Deterministic."""
from __future__ import annotations

import numpy as np


def make_classification(n: int = 400, d: int = 5, *, seed: int = 0
                        ) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1, d)
    X = rng.normal(0, 1, (n, d))
    logits = X @ w
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n) < p).astype(int)
    return X, y
