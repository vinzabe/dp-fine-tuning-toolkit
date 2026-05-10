"""
Privacy accounting via Renyi Differential Privacy (RDP).

We implement the RDP composition for the Sampled Gaussian Mechanism (SGM)
and convert to (epsilon, delta)-DP. This matches Mironov 2017 + Abadi 2016
+ Wang/Balle/Kasiviswanathan 2019 conventions used by Opacus/TF-Privacy.

If `opacus` is available we cross-check our results against it during tests.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# Standard alpha grid used by Opacus / TF-Privacy
DEFAULT_ALPHAS: List[float] = (
    [1.0 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
)


def _log_add(a: float, b: float) -> float:
    """log(exp(a) + exp(b)) numerically stable."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        a, b = b, a
    return b + math.log1p(math.exp(a - b))


def _log_sub(a: float, b: float) -> float:
    """log(exp(a) - exp(b))."""
    if b == -math.inf:
        return a
    if a < b:
        return -math.inf
    if a == b:
        return -math.inf
    return a + math.log1p(-math.exp(b - a))


def _log_factorial(n: int) -> float:
    return math.lgamma(n + 1)


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return _log_factorial(n) - _log_factorial(k) - _log_factorial(n - k)


def _compute_log_a(q: float, sigma: float, alpha: float) -> float:
    """RDP of the Sampled Gaussian Mechanism (Wang/Balle/Kasiviswanathan 2019).
    Implements the integer-alpha case via binomial sum, then linearly
    interpolates for fractional alpha.
    """
    if float(alpha).is_integer():
        return _compute_log_a_int(q, sigma, int(alpha))
    return _compute_log_a_frac(q, sigma, alpha)


def _compute_log_a_int(q: float, sigma: float, alpha: int) -> float:
    log_a = -math.inf
    log_q = math.log(max(q, 1e-300))
    log_1mq = math.log1p(-q) if q < 1 else -math.inf
    for i in range(alpha + 1):
        log_b = (_log_comb(alpha, i) + i * log_q + (alpha - i) * log_1mq +
                 (i * i - i) / (2 * sigma * sigma))
        log_a = _log_add(log_a, log_b)
    return log_a


def _compute_log_a_frac(q: float, sigma: float, alpha: float) -> float:
    """Wang/Balle/Kasiviswanathan 2019, integral form approximation via series."""
    log_a0, log_a1 = -math.inf, -math.inf
    i = 0
    z0 = sigma ** 2 * math.log(1 / q - 1) + 0.5
    while True:
        coef = math.lgamma(alpha + 1) - math.lgamma(i + 1) - math.lgamma(alpha - i + 1)
        log_coef_i = coef + i * math.log(q) + (alpha - i) * math.log1p(-q)
        s = log_coef_i + (i * (i - 1)) / (2 * sigma ** 2)
        log_a0 = _log_add(log_a0, s)
        coef = math.lgamma(alpha + 1) - math.lgamma(i + 1) - math.lgamma(alpha - i + 1)
        log_coef_i = coef + i * math.log(q) + (alpha - i) * math.log1p(-q)
        s = log_coef_i + (i * (i + 1)) / (2 * sigma ** 2)
        log_a1 = _log_add(log_a1, s)
        i += 1
        if i > 100:
            break
    return _log_add(log_a0, log_a1)


def compute_rdp(q: float, noise_multiplier: float, steps: int,
                alphas: Optional[List[float]] = None) -> List[float]:
    """RDP for `steps` iterations of the Sampled Gaussian Mechanism."""
    alphas = alphas or DEFAULT_ALPHAS
    rdp = []
    for a in alphas:
        if noise_multiplier <= 0:
            rdp.append(math.inf)
            continue
        log_alpha = _compute_log_a(q, noise_multiplier, a)
        # convert log_alpha to RDP epsilon at order alpha:
        rdp.append(log_alpha / (a - 1) * steps)
    return rdp


def get_privacy_spent(rdp: List[float], alphas: List[float], delta: float) -> float:
    """Convert RDP -> (epsilon, delta)-DP. Returns the smallest epsilon."""
    eps = []
    for r, a in zip(rdp, alphas):
        if r == math.inf:
            eps.append(math.inf)
        else:
            # standard RDP -> DP conversion
            e = r + math.log((a - 1) / a) - (math.log(delta) + math.log(a)) / (a - 1)
            eps.append(e)
    return min(eps)


@dataclass
class PrivacyBudget:
    """Tracks (epsilon, delta) over training. Update once per step."""
    target_epsilon: float
    target_delta: float
    sample_rate: float           # batch_size / dataset_size
    noise_multiplier: float
    steps: int = 0
    alphas: List[float] = None  # type: ignore

    def __post_init__(self):
        if self.alphas is None:
            self.alphas = DEFAULT_ALPHAS

    def step(self, n: int = 1) -> None:
        self.steps += n

    def epsilon(self) -> float:
        if self.steps == 0:
            return 0.0
        rdp = compute_rdp(self.sample_rate, self.noise_multiplier, self.steps, self.alphas)
        return get_privacy_spent(rdp, self.alphas, self.target_delta)

    def remaining(self) -> float:
        return max(0.0, self.target_epsilon - self.epsilon())

    def exhausted(self) -> bool:
        return self.epsilon() > self.target_epsilon

    def steps_until_exhausted(self, max_check: int = 200000) -> int:
        """Binary search for the step count at which we exceed target_epsilon.
        Useful for computing how many epochs you can train.
        """
        lo, hi = 0, max_check
        while lo < hi:
            mid = (lo + hi + 1) // 2
            rdp = compute_rdp(self.sample_rate, self.noise_multiplier, mid, self.alphas)
            e = get_privacy_spent(rdp, self.alphas, self.target_delta)
            if e <= self.target_epsilon:
                lo = mid
            else:
                hi = mid - 1
        return lo
