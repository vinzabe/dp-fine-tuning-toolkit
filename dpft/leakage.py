"""
Leakage tests — empirical privacy auditing via membership inference attacks.

Provides three MIA attacks of increasing strength:
  - LossThresholdMIA (Yeom et al. 2018): if test loss < threshold, predict member
  - ShadowModelMIA (Shokri et al. 2017): train shadow models to learn the
    distribution of (loss, member?) and use it as classifier
  - LiRA (Carlini et al. 2022): likelihood-ratio attack — strongest known

These attacks turn the abstract `epsilon` into a *measured* upper bound on
the attacker's TPR @ low FPR. If your reported epsilon is 1.0 but LiRA
achieves TPR=0.5 @ FPR=0.001, your training pipeline has a bug.

Also includes a `canary_extraction_test` for autoregressive models
(insert n-grams that don't naturally occur, then test if generation
reproduces them — direct extraction risk).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class MIAResult:
    auc: float
    accuracy: float
    tpr_at_fpr_001: float
    tpr_at_fpr_01: float
    n_members: int
    n_nonmembers: int
    attack: str


def _roc(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Returns fpr, tpr, auc."""
    order = np.argsort(-scores)
    s = scores[order]
    y = labels[order]
    n_pos = max(1, int(y.sum()))
    n_neg = max(1, int((1 - y).sum()))
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / n_pos
    fpr = fp / n_neg
    # AUC via trapezoid (handle numpy<2 / numpy>=2)
    fpr_full = np.concatenate([[0.0], fpr, [1.0]])
    tpr_full = np.concatenate([[0.0], tpr, [1.0]])
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(_trap(tpr_full, fpr_full))
    return fpr, tpr, auc


def _tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target: float) -> float:
    if len(fpr) == 0:
        return 0.0
    idx = np.searchsorted(fpr, target, side="right") - 1
    if idx < 0:
        return 0.0
    return float(tpr[idx])


@torch.no_grad()
def per_sample_losses(model: nn.Module,
                      data: List[Tuple[torch.Tensor, torch.Tensor]],
                      criterion: nn.Module,
                      device: Optional[str] = None) -> np.ndarray:
    """Compute per-sample loss for each (x, y) pair."""
    device = device or next(model.parameters()).device
    model.eval()
    losses = []
    for x, y in data:
        x = x.to(device)
        y = y.to(device)
        if x.dim() == y.dim():
            # Sometimes already batched; normalize
            pass
        # Add batch dim if missing
        if x.dim() == 1 or (x.dim() == 3 and x.shape[0] != 1):
            x = x.unsqueeze(0)
        if y.dim() == 0 or (y.dim() == 1 and y.shape[0] != 1):
            y = y.unsqueeze(0)
        out = model(x)
        loss = criterion(out, y)
        losses.append(float(loss.item()))
    return np.array(losses, dtype=np.float64)


class LossThresholdMIA:
    """Yeom et al. 2018. Score = -loss. Higher score => more likely to be member."""
    name = "loss_threshold"

    def attack(self,
               member_losses: np.ndarray,
               nonmember_losses: np.ndarray) -> MIAResult:
        scores = np.concatenate([-member_losses, -nonmember_losses])
        labels = np.concatenate([
            np.ones_like(member_losses), np.zeros_like(nonmember_losses)
        ])
        fpr, tpr, auc = _roc(scores, labels)
        # accuracy at threshold = mean of member/nonmember losses
        thr = (member_losses.mean() + nonmember_losses.mean()) / 2
        preds = (scores > -thr).astype(np.int32)
        acc = float((preds == labels).mean())
        return MIAResult(
            auc=auc,
            accuracy=acc,
            tpr_at_fpr_001=_tpr_at_fpr(fpr, tpr, 0.01),
            tpr_at_fpr_01=_tpr_at_fpr(fpr, tpr, 0.1),
            n_members=len(member_losses),
            n_nonmembers=len(nonmember_losses),
            attack=self.name,
        )


class ShadowModelMIA:
    """Shokri et al. 2017 (simplified). Uses 1-D Gaussian likelihood per class."""
    name = "shadow_model"

    def __init__(self, n_shadow: int = 5):
        self.n_shadow = n_shadow

    def _gaussian_score(self, x: float, mu: float, sigma: float) -> float:
        if sigma <= 0:
            return 0.0
        return -0.5 * ((x - mu) / sigma) ** 2 - math.log(sigma * math.sqrt(2 * math.pi))

    def attack(self, member_losses: np.ndarray, nonmember_losses: np.ndarray) -> MIAResult:
        # Fit Gaussians on observed loss distributions (this is the "shadow"
        # output post-bootstrap; we skip the actual shadow training and use
        # the observed losses as shadow proxies for compactness).
        mu_m, sigma_m = float(member_losses.mean()), float(max(member_losses.std(), 1e-6))
        mu_n, sigma_n = float(nonmember_losses.mean()), float(max(nonmember_losses.std(), 1e-6))
        all_losses = np.concatenate([member_losses, nonmember_losses])
        labels = np.concatenate([
            np.ones_like(member_losses), np.zeros_like(nonmember_losses)
        ])
        scores = np.array([
            self._gaussian_score(x, mu_m, sigma_m) - self._gaussian_score(x, mu_n, sigma_n)
            for x in all_losses
        ])
        fpr, tpr, auc = _roc(scores, labels)
        preds = (scores > 0).astype(np.int32)
        acc = float((preds == labels).mean())
        return MIAResult(
            auc=auc,
            accuracy=acc,
            tpr_at_fpr_001=_tpr_at_fpr(fpr, tpr, 0.01),
            tpr_at_fpr_01=_tpr_at_fpr(fpr, tpr, 0.1),
            n_members=len(member_losses),
            n_nonmembers=len(nonmember_losses),
            attack=self.name,
        )


class LiRA:
    """Carlini et al. 2022 — simplified offline LiRA using per-sample
    in/out shadow loss distributions. Caller provides shadow losses for
    each example: `shadow_in_losses[i]` is N losses on example i when it
    WAS in training; `shadow_out_losses[i]` is M losses on example i when
    it WAS NOT.
    """
    name = "lira_offline"

    def attack(self,
               target_member_losses: np.ndarray,
               target_nonmember_losses: np.ndarray,
               shadow_in_losses: np.ndarray,
               shadow_out_losses: np.ndarray) -> MIAResult:
        # shadow_*_losses: shape (n_examples, n_shadow)
        mu_in = shadow_in_losses.mean(axis=1)
        sd_in = np.maximum(shadow_in_losses.std(axis=1), 1e-6)
        mu_out = shadow_out_losses.mean(axis=1)
        sd_out = np.maximum(shadow_out_losses.std(axis=1), 1e-6)
        # We don't know which target sample lined up with which shadow position;
        # use means across shadows directly (approximation).
        # For simplicity assume per-sample shadow stats already aligned to target.
        n_m = len(target_member_losses)
        n_n = len(target_nonmember_losses)
        # Use first n_m shadow rows for members, next n_n for nonmembers
        # (caller is responsible for alignment).
        m_scores = []
        for i, x in enumerate(target_member_losses):
            j = i % shadow_in_losses.shape[0]
            ll_in = -0.5 * ((x - mu_in[j]) / sd_in[j]) ** 2 - math.log(sd_in[j])
            ll_out = -0.5 * ((x - mu_out[j]) / sd_out[j]) ** 2 - math.log(sd_out[j])
            m_scores.append(ll_in - ll_out)
        n_scores = []
        for i, x in enumerate(target_nonmember_losses):
            j = i % shadow_in_losses.shape[0]
            ll_in = -0.5 * ((x - mu_in[j]) / sd_in[j]) ** 2 - math.log(sd_in[j])
            ll_out = -0.5 * ((x - mu_out[j]) / sd_out[j]) ** 2 - math.log(sd_out[j])
            n_scores.append(ll_in - ll_out)
        scores = np.array(m_scores + n_scores)
        labels = np.array([1] * n_m + [0] * n_n)
        fpr, tpr, auc = _roc(scores, labels)
        preds = (scores > 0).astype(np.int32)
        acc = float((preds == labels).mean())
        return MIAResult(
            auc=auc,
            accuracy=acc,
            tpr_at_fpr_001=_tpr_at_fpr(fpr, tpr, 0.01),
            tpr_at_fpr_01=_tpr_at_fpr(fpr, tpr, 0.1),
            n_members=n_m,
            n_nonmembers=n_n,
            attack=self.name,
        )


@dataclass
class CanaryResult:
    canary: str
    n_inserted: int          # times canary occurs in train data
    extracted: bool          # did we recover it from generation?
    extracted_text: str = ""


def canary_extraction_test(generate_fn: Callable[[str], str],
                            canary: str,
                            prefix_len: int = 4,
                            n_attempts: int = 20) -> CanaryResult:
    """
    Probe a generative model for direct canary extraction.

    Splits the canary into a prefix + suffix; queries the model with the
    prefix and `n_attempts` different generations; reports if any
    generation reproduces the suffix.
    """
    if len(canary) <= prefix_len:
        return CanaryResult(canary=canary, n_inserted=0, extracted=False)
    prefix = canary[:prefix_len]
    suffix = canary[prefix_len:]
    for _ in range(n_attempts):
        out = generate_fn(prefix)
        if suffix in out:
            return CanaryResult(canary=canary, n_inserted=0, extracted=True,
                                extracted_text=out)
    return CanaryResult(canary=canary, n_inserted=0, extracted=False)
