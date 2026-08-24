# 3. Implement DP-SGD and the accountant in visible numpy, not a framework

Date: 2026-08-24
Status: Accepted

## Context
Opacity is how DP bugs hide. A framework optimizer that "does DP-SGD" is a box the
reader cannot inspect, and the whole value of this project is that the privacy
machinery is checkable.

## Decision
Implement DP-SGD (per-example clipping + Gaussian noise on logistic regression)
and the RDP accountant directly in numpy. Deterministic given a seed.

## Consequences
- Every privacy-relevant step — per-example gradients, the L2 clip, the noise
  scale (σ·C), Poisson subsampling — is a few readable lines, so a reviewer can
  confirm the mechanism matches what the accountant assumes.
- Determinism makes the audit reproducible and the trade-off tests stable.
- The model is intentionally simple (logistic regression). The mechanism and audit
  generalize to any per-example-gradient model; the toolkit demonstrates the method
  rather than shipping a production deep-learning stack. Stated as a non-goal.
