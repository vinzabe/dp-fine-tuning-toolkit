# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in **dp-fine-tuning-toolkit**,
please report it privately. Do **not** open a public GitHub issue.

**Email:** security@vinzabe.dev (or open a GitHub Security Advisory)

Please include:
- A clear description of the issue
- Steps to reproduce (PoC preferred)
- The version / commit SHA you tested against
- Any suggested mitigation

We aim to acknowledge new reports within **72 hours** and to publish a
fix or mitigation within **30 days** for high-severity issues.

## Scope

In scope (privacy-correctness bugs):
- The privacy accountant (`accountant.py`) reports an `epsilon` that
  is **lower** than the true privacy loss for the given (q, sigma, T)
  configuration
- The DP-SGD trainer (`trainer.py`) skips per-sample clipping or
  Gaussian noise on any code path
- The Opacus integration silently downgrades to non-DP training
- Privacy-budget tracking does not halt training when the configured
  `target_epsilon` is reached
- Membership-inference helpers (`leakage.py`) leak training-set indices
  through their public API beyond the documented oracle access

Out of scope:
- Inherent limits of (eps, delta)-DP (e.g. very small eps still allows
  some membership inference at high attacker advantage) — report these
  as research findings, not bugs
- Numerical-precision artefacts at extreme parameter ranges (sigma
  < 0.1 or > 100) — these regimes are not supported

## Threat model

A model trainer wants to fine-tune on private data and prove a
**worst-case** bound on per-record privacy loss to a regulator,
auditor, or downstream user. We assume:

- The trainer is **honest**: it correctly reports the (eps, delta) it
  achieved; this toolkit's job is to make that claim accurate.
- An adversary has **black-box** access to the released model
  (typical) or **white-box** if the model is published (worst case).
- The training data set is treated as a single sensitive table;
  protecting *one record* with (eps, delta)-DP via the Sampled
  Gaussian Mechanism is the explicit guarantee.
- Opacus is used when available; the manual path is byte-for-byte
  privacy-equivalent (per-sample-clip + sum + Gaussian noise) for
  validation and in environments where Opacus is unavailable.

This toolkit does **not** protect against:
- Side channels (timing, memory, page faults) leaking training data
- Adversaries who can submit poisoned training data (use FL defenses
  from the federated-attack-lab project for that)
- Compromise of the trainer's machine

## Operational guidance

- Always set `target_epsilon` and `target_delta` BEFORE training; the
  toolkit will halt as soon as the budget is exhausted.
- Verify accountant output with at least one independent implementation
  (e.g. Opacus' `RDPAccountant`) — `tests/test_dpft.py` does this.
- Choose `delta < 1 / N` where N is the dataset size.
- Use `sample_rate = batch_size / dataset_size`; do NOT confuse this
  with an iteration ratio.
- Audit the privacy claim post-training by running the included MIA
  attacks (`leakage.LossThresholdMIA`, `LiRA`) — `dp_sgd_reduces_mia_auc`
  test ships an end-to-end demonstration.

## Supply chain

- All Python deps pinned via `requirements.txt`
- `opacus>=1.5` provides the validated cross-check accountant
- Shared `llm_client.py` vendored (not a third-party package)
