# 2. Audit the privacy guarantee empirically, do not merely account for it

Date: 2026-08-24
Status: Accepted

## Context
An RDP accountant computes ε from the mechanism's parameters. But the accountant
describes the mechanism you *intended*, not the code you *ran*. The most common DP
failures — clipping the summed gradient instead of per-example, noising with the
wrong scale, miscounting steps or the sampling rate — leave the accountant's ε
untouched while the real privacy loss balloons. An unaudited ε is a statement
about the math, not about the software.

## Decision
Ship an empirical audit alongside the accountant: a membership-inference attack
that produces a statistical lower bound on ε from the attack's true/false positive
rates, and assert `empirical_epsilon <= theoretical_epsilon`. A violation fails the
CLI with exit code 2.

## Consequences
- The toolkit catches implementation bugs the accountant cannot see; the audit and
  the mechanism are tested together (`test_audit_consistent_for_correct_mechanism`).
- The audit is a LOWER bound: passing means "no attacker in this test did better
  than the claim", not "ε is exactly right". A tighter audit needs more trials and
  a stronger attack — documented as the resolution limit.
- The empirical→ε conversion is validated on synthetic (TPR,FPR) points: chance
  performance yields ~0, and a larger attack advantage yields a larger ε
  (`test_empirical_epsilon_*`).
- Cost: the audit trains many models, so it is slower than accounting. That is the
  price of a checkable guarantee, and the trial count is tunable.
