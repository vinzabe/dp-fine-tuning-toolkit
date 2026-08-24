"""dpft — differentially private training whose privacy claim is audited, not asserted.

Differential privacy has a credibility problem: papers report an epsilon computed
by an accountant, but accounting bugs are common and a wrong clip or a
miscomposed step silently inflates the real privacy loss. A number nobody can
check is a number nobody should trust.

This toolkit does three things and ties them together:

  1. **DP-SGD** — per-example gradient clipping (bounds sensitivity) plus Gaussian
     noise (the mechanism).
  2. **An RDP accountant** for the subsampled Gaussian mechanism, converting to
     (epsilon, delta).
  3. **An empirical privacy audit** — a membership-inference attack that produces a
     statistical LOWER BOUND on epsilon. If the empirical lower bound ever exceeds
     the theoretical epsilon, the guarantee is violated and the audit fails loudly.

The audit is the point. It is the difference between "we computed epsilon = 2.1"
and "we computed 2.1 and an attacker could not do better than an empirical 0.9,
which is consistent with the claim."
"""
__version__ = "1.0.0"
