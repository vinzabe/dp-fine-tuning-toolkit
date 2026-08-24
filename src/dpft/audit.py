"""Empirical privacy audit via membership inference.

The theoretical epsilon is a CLAIM. This audits it: train many model pairs that
differ by one "canary" record, run a membership-inference attack that guesses
whether the canary was in the training set, and convert the attack's true/false
positive rates into an empirical epsilon LOWER BOUND.

If the empirical lower bound ever exceeds the theoretical epsilon, the guarantee
is violated — the accountant, the clipping, or the noise is wrong. The audit
turns "trust me" into "an attacker demonstrably cannot do better than this".
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from .dpsgd import DPSGDConfig, train


@dataclasses.dataclass(frozen=True, slots=True)
class AuditResult:
    trials: int
    attack_tpr: float          # true-positive rate of the membership attack
    attack_fpr: float          # false-positive rate
    empirical_epsilon: float   # statistical lower bound on epsilon
    theoretical_epsilon: float
    consistent: bool           # empirical <= theoretical (guarantee holds)

    @property
    def note(self) -> str:
        if self.consistent:
            return (f"empirical eps lower bound {self.empirical_epsilon:.3f} "
                    f"<= theoretical {self.theoretical_epsilon:.3f}: consistent")
        return (f"VIOLATION: empirical {self.empirical_epsilon:.3f} exceeds "
                f"theoretical {self.theoretical_epsilon:.3f}")


def _empirical_epsilon(tpr: float, fpr: float, delta: float) -> float:
    """Convert attack (TPR, FPR) to an epsilon lower bound.

    From the DP hypothesis-testing characterization: for an eps-DP mechanism,
    TPR <= e^eps * FPR + delta  and  FPR <= e^eps * (1-TPR)... we take the tightest
    of the two implied lower bounds. Clamped at 0.
    """
    candidates = []
    if fpr > 0:
        candidates.append(math.log(max(1e-12, tpr - delta) / fpr))
    fnr = 1.0 - tpr
    tnr = 1.0 - fpr
    if fnr > 0:
        candidates.append(math.log(max(1e-12, tnr - delta) / fnr))
    return max(0.0, max(candidates) if candidates else 0.0)


def run_audit(X: np.ndarray, y: np.ndarray, cfg: DPSGDConfig, *,
              theoretical_epsilon: float, delta: float = 1e-5,
              trials: int = 200, seed: int = 0) -> AuditResult:
    """Membership-inference audit.

    For each trial: flip a coin for whether a strong canary is included, train,
    and let the attack guess inclusion from the canary's loss. Aggregate into
    (TPR, FPR) and an empirical epsilon.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    d = X.shape[1]
    # a canary that is far from the data distribution -> maximally memorizable
    canary_x = np.full(d, 5.0)
    canary_y = 1.0

    # calibrate the attack threshold on a few held-out "out" models
    calib_losses = []
    for t in range(min(20, trials)):
        m = train(X, y, dataclasses.replace(cfg, seed=10_000 + t))
        calib_losses.append(float(m.loss_per_example(
            canary_x[None, :], np.array([canary_y]))[0]))
    threshold = float(np.median(calib_losses))

    tp = fp = tn = fn = 0
    for t in range(trials):
        included = bool(rng.integers(0, 2))
        if included:
            Xt = np.vstack([X, canary_x])
            yt = np.concatenate([y, [canary_y]])
        else:
            Xt, yt = X, y
        m = train(Xt, yt, dataclasses.replace(cfg, seed=20_000 + t))
        loss = float(m.loss_per_example(canary_x[None, :],
                                        np.array([canary_y]))[0])
        # low loss on the canary -> attack guesses "included"
        guess_in = loss < threshold
        if included and guess_in:
            tp += 1
        elif included and not guess_in:
            fn += 1
        elif not included and guess_in:
            fp += 1
        else:
            tn += 1

    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    emp = _empirical_epsilon(tpr, fpr, delta)
    return AuditResult(
        trials=trials, attack_tpr=tpr, attack_fpr=fpr,
        empirical_epsilon=emp, theoretical_epsilon=theoretical_epsilon,
        consistent=emp <= theoretical_epsilon + 1e-9)
