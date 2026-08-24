"""Rényi Differential Privacy accountant for the subsampled Gaussian mechanism.

Standard, well-defined math (Mironov 2017; Wang et al. subsampled RDP). We compute
RDP at a set of orders for one subsampled-Gaussian step, compose linearly over
steps, and convert the tightest order to (epsilon, delta). Pure numpy.
"""
from __future__ import annotations

import dataclasses
import math

# Orders to search. More orders = tighter epsilon; these are the standard set.
DEFAULT_ORDERS: tuple[float, ...] = tuple(
    [1.0 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)))


def _log_add(a: float, b: float) -> float:
    """log(exp(a) + exp(b)) stably."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


def _rdp_gaussian(alpha: float, sigma: float) -> float:
    """RDP of the (non-subsampled) Gaussian mechanism at order alpha."""
    return alpha / (2.0 * sigma * sigma)


def _rdp_subsampled_gaussian(alpha: float, q: float, sigma: float) -> float:
    """RDP at INTEGER order for Poisson-subsampled Gaussian, via the binomial
    expansion bound (Wang, Balle, Kasiviswanathan). alpha must be an integer >= 2.
    """
    if q == 0.0:
        return 0.0
    if q == 1.0:
        return _rdp_gaussian(alpha, sigma)
    ia = int(alpha)
    log_terms = []
    for k in range(ia + 1):
        log_coef = (math.lgamma(ia + 1) - math.lgamma(k + 1)
                    - math.lgamma(ia - k + 1))
        term = (log_coef + k * math.log(q) + (ia - k) * math.log1p(-q)
                + (k * k - k) / (2.0 * sigma * sigma))
        log_terms.append(term)
    log_sum = -math.inf
    for t in log_terms:
        log_sum = _log_add(log_sum, t)
    return log_sum / (alpha - 1.0)


@dataclasses.dataclass(frozen=True, slots=True)
class PrivacySpent:
    epsilon: float
    delta: float
    best_order: float
    steps: int
    noise_multiplier: float
    sampling_rate: float


def compute_epsilon(*, noise_multiplier: float, sampling_rate: float, steps: int,
                    delta: float = 1e-5,
                    orders: tuple[float, ...] = DEFAULT_ORDERS) -> PrivacySpent:
    """(epsilon, delta) after `steps` of subsampled-Gaussian DP-SGD."""
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be > 0")
    if not 0 < sampling_rate <= 1:
        raise ValueError("sampling_rate must be in (0, 1]")
    if steps < 0:
        raise ValueError("steps must be >= 0")
    if steps == 0:
        # No queries means no privacy loss, regardless of the order search.
        return PrivacySpent(epsilon=0.0, delta=delta, best_order=orders[0],
                            steps=0, noise_multiplier=noise_multiplier,
                            sampling_rate=sampling_rate)

    best_eps = math.inf
    best_order = orders[0]
    for alpha in orders:
        if alpha <= 1.0:
            continue
        ia = int(round(alpha))
        if abs(alpha - ia) < 1e-9 and ia >= 2:
            rdp_step = _rdp_subsampled_gaussian(float(ia), sampling_rate,
                                                noise_multiplier)
        else:
            # non-integer order: use the (looser) non-subsampled bound scaled by q^2
            rdp_step = (sampling_rate ** 2) * _rdp_gaussian(alpha, noise_multiplier)
        rdp = rdp_step * steps
        # RDP -> (eps, delta) conversion (Balle et al. tight form)
        eps = rdp + math.log1p(-1.0 / alpha) - math.log(delta * alpha) / (alpha - 1.0)
        eps = max(eps, rdp - math.log(delta * alpha) / (alpha - 1.0))
        if eps < best_eps:
            best_eps = eps
            best_order = alpha
    return PrivacySpent(
        epsilon=max(0.0, best_eps), delta=delta, best_order=best_order,
        steps=steps, noise_multiplier=noise_multiplier,
        sampling_rate=sampling_rate)


def noise_for_epsilon(target_epsilon: float, *, sampling_rate: float, steps: int,
                      delta: float = 1e-5) -> float:
    """Binary-search the noise multiplier that achieves `target_epsilon`."""
    lo, hi = 0.1, 100.0
    for _ in range(60):
        mid = (lo + hi) / 2
        eps = compute_epsilon(noise_multiplier=mid, sampling_rate=sampling_rate,
                              steps=steps, delta=delta).epsilon
        if eps > target_epsilon:
            lo = mid       # too much privacy loss -> need more noise
        else:
            hi = mid
    return hi
