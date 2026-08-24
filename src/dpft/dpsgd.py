"""DP-SGD: per-example gradient clipping + Gaussian noise.

Two ingredients make SGD differentially private:
  * **Clip** each per-example gradient to L2 norm C. This bounds one example's
    influence — the sensitivity — regardless of how much of an outlier it is.
  * **Add Gaussian noise** with std = sigma * C to the summed gradients. This is
    the mechanism whose privacy the accountant measures.

Implemented on logistic regression with numpy so the mechanism is fully visible
and deterministic given a seed — not hidden inside a framework optimizer.
"""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(slots=True)
class DPSGDConfig:
    clip_norm: float = 1.0          # C: per-example gradient L2 clip
    noise_multiplier: float = 1.0   # sigma: noise std = sigma * C
    lot_size: int = 32              # expected batch (Poisson) size
    lr: float = 0.1
    epochs: int = 20
    seed: int = 0

    def __post_init__(self) -> None:
        if self.clip_norm <= 0:
            raise ValueError("clip_norm must be > 0")
        if self.noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be > 0")
        if self.lot_size < 1:
            raise ValueError("lot_size must be >= 1")


@dataclasses.dataclass(slots=True)
class TrainedModel:
    weights: np.ndarray
    bias: float
    steps: int
    sampling_rate: float

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = np.asarray(X, dtype=float) @ self.weights + self.bias
        result = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
        return np.asarray(result, dtype=float)

    def loss_per_example(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        p = np.clip(self.predict_proba(X), 1e-12, 1 - 1e-12)
        y = np.asarray(y, dtype=float)
        loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        return np.asarray(loss, dtype=float)


def _per_example_grads(w: np.ndarray, b: float, X: np.ndarray,
                       y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = X @ w + b
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    err = p - y                        # (n,)
    grad_w = err[:, None] * X          # (n, d) per-example weight grads
    grad_b = err                        # (n,) per-example bias grads
    return grad_w, grad_b


def _clip_rows(grads: np.ndarray, clip: float) -> np.ndarray:
    norms = np.linalg.norm(grads, axis=1, keepdims=True)
    factor = np.minimum(1.0, clip / (norms + 1e-12))
    return np.asarray(grads * factor, dtype=float)


def train(X: np.ndarray, y: np.ndarray, cfg: DPSGDConfig) -> TrainedModel:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    rng = np.random.default_rng(cfg.seed)
    w = np.zeros(d)
    b = 0.0
    q = min(1.0, cfg.lot_size / n)
    steps = 0
    steps_per_epoch = max(1, n // cfg.lot_size)

    for _ in range(cfg.epochs):
        for _ in range(steps_per_epoch):
            # Poisson subsampling: each example included independently w.p. q
            mask = rng.random(n) < q
            idx = np.nonzero(mask)[0]
            if len(idx) == 0:
                steps += 1
                continue
            gw, gb = _per_example_grads(w, b, X[idx], y[idx])
            # clip per example on the COMBINED (w,b) gradient vector
            combined = np.concatenate([gw, gb[:, None]], axis=1)
            clipped = _clip_rows(combined, cfg.clip_norm)
            summed = clipped.sum(axis=0)
            # add Gaussian noise scaled by sigma * C
            noise = rng.normal(0.0, cfg.noise_multiplier * cfg.clip_norm,
                               size=summed.shape)
            noisy = (summed + noise) / max(1, len(idx))
            w -= cfg.lr * noisy[:d]
            b -= cfg.lr * float(noisy[d])
            steps += 1

    return TrainedModel(weights=w, bias=b, steps=steps, sampling_rate=q)
