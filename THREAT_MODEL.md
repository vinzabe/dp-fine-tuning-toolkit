# Threat model & scope

## What this is
A toolkit for training with differential privacy AND empirically validating the
resulting guarantee. It is a methodology/education-grade implementation with a real,
checkable audit — not a drop-in replacement for a hardened DP training framework.

## The guarantee, precisely
- **Theoretical ε** comes from an RDP accountant for the subsampled Gaussian
  mechanism. It is correct for the mechanism as implemented here (per-example clip,
  σ·C Gaussian noise, Poisson subsampling), which the audit checks.
- **Empirical ε** is a statistical LOWER bound from membership inference. "Audit
  consistent" means no attacker in the test exceeded the theoretical ε — it is
  evidence for the claim, not a proof of it. More trials + a stronger attack give a
  tighter bound.

## Trust boundaries & assumptions
- **Determinism relies on the seed and on float reproducibility.** Cross-platform
  float differences can perturb the audit slightly; trial counts are chosen with
  margin.
- **Poisson subsampling is assumed** by the accountant and implemented by the
  trainer; using a different sampler invalidates the ε. This coupling is
  intentional and tested.

## Non-goals (stated plainly)
- **Not a production DP framework.** No secure RNG for the noise (uses numpy's PRNG
  — a real deployment needs a cryptographically secure source), no distributed
  training, no deep networks. These are out of scope and would each be a project.
- **Not a proof.** The audit bounds ε from below empirically; it does not certify
  the upper bound. Treat a consistent audit as strong evidence, not a guarantee.
- **δ and the accountant's tightness** follow standard bounds; a specialized
  accountant may give a tighter ε. Documented, not hidden.

## Reporting
An audit that is *inconsistent* on the shipped mechanism, or an accountant result
that disagrees with a reference (e.g. Opacus) beyond tolerance, is a correctness
bug — report to **gabejar@usa.com** with parameters.
