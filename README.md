# dp-fine-tuning-toolkit

> Differentially-private fine-tuning with **DP-SGD**, an **independent privacy accountant**, and an end-to-end **MIA / canary-extraction** auditor that proves the budget actually buys you something.

[![tests](https://img.shields.io/badge/tests-25%2F25-brightgreen)](#tests)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

This toolkit packages everything needed to fine-tune a PyTorch model
under (ε, δ)-Differential Privacy and **prove** the privacy claim with
real attacks:

- A from-scratch **RDP accountant** for the Sampled Gaussian Mechanism
  cross-checked against Opacus' `RDPAccountant` (within 30%).
- A `DPSGDTrainer` with two backends:
  1. **Opacus** (`make_private` path) when installed
  2. **Manual** per-sample-clip + sum + Gaussian noise loop for
     environments where Opacus is unavailable, byte-equivalent on the
     privacy side.
- A **leakage** module shipping four membership-inference attacks
  (`LossThresholdMIA`, `ShadowModelMIA`, `LiRA` offline, plus a
  canary-extraction probe) that demonstrate the privacy claim end-to-end.
- A `dpft` CLI that reports your remaining privacy budget, the steps
  remaining at a given (q, σ, target_ε), and audits a trained model.

## Install

```bash
git clone https://github.com/vinzabe/dp-fine-tuning-toolkit
cd dp-fine-tuning-toolkit
pip install -r requirements.txt
```

`opacus` is optional but recommended. Without it the trainer falls back
to the manual path.

## Quickstart

### Privacy budget calculator

```bash
python -m dpft.cli budget --q 0.01 --sigma 1.0 --steps 1000
# eps=2.108  delta=1e-5  alpha*=4.0
```

### Steps until budget exhausted

```bash
python -m dpft.cli steps --q 0.01 --sigma 1.0 --target-eps 4.0
# 3835 steps remaining
```

### Train under DP-SGD (programmatic)

```python
from dpft import DPSGDTrainer, DPTrainingArgs, make_compute_loss_fn
import torch.nn as nn

args = DPTrainingArgs(
    learning_rate=0.05,
    num_train_epochs=2,
    per_device_batch_size=32,
    max_grad_norm=1.0,
    noise_multiplier=2.0,
    target_epsilon=10.0,
    target_delta=1e-5,
    use_opacus=True,
)
trainer = DPSGDTrainer(model, train_loader, args, make_compute_loss_fn(nn.CrossEntropyLoss()))
metrics = trainer.train()
print(metrics.epsilon_spent, metrics.history[-1])
```

### Audit a trained model

```bash
python -m dpft.cli audit --model checkpoint.pt --shadow shadow.pt
# loss-thr  AUC=0.58  TPR@FPR=1%=0.05
# LiRA      AUC=0.55  TPR@FPR=1%=0.04
# canary memorization detected: NO
```

## Privacy claim, demonstrated

The end-to-end `test_dp_actually_reduces_leakage` test trains the same
model twice on the same overfit-prone setup, once without DP and once
with DP-SGD, then runs the loss-threshold MIA against both:

| run        | mean train loss | mean test loss | MIA AUC |
|------------|----------------:|---------------:|--------:|
| no-DP      |            0.15 |           1.32 |   0.872 |
| DP-SGD     |            0.72 |           0.84 |   0.583 |

DP-SGD pulls the MIA AUC from **0.87 → 0.58** (close to the chance
level of 0.5).

## Architecture

```
                    ┌──────────────────────────┐
                    │ accountant.py            │
                    │   RDP for SGM            │
   (q, σ, T) ─────▶ │   alpha-search → (ε, δ) │ ─▶ PrivacyBudget
                    └──────────────────────────┘

                    ┌──────────────────────────────────────────┐
   model + data ──▶ │ trainer.py                                │
                    │  Opacus path: PrivacyEngine.make_private  │
                    │  Manual path: per-sample clip + sum +     │
                    │                Gaussian noise + step      │
                    │  StepHistory + epsilon_spent              │
                    └──────────────────────────────────────────┘

                    ┌──────────────────────────────────────────┐
   trained model ──▶│ leakage.py                                │
                    │  LossThresholdMIA / ShadowModelMIA        │
                    │  LiRA (offline) / canary_extraction_test  │
                    │  → AUC, TPR@FPR=1%, TPR@FPR=10%           │
                    └──────────────────────────────────────────┘
```

## Tests

```bash
python tests/test_dpft.py
```

Snapshot:

```
DP Fine-Tuning: 25 passed, 0 failed
  - Accountant cross-checks Opacus within 30%
  - Manual path matches Opacus epsilon trajectory
  - DP-SGD reduces MIA AUC from 0.87 -> 0.58 end-to-end
```

## License

MIT — see [LICENSE](LICENSE).
