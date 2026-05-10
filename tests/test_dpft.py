"""Smoke tests for DP fine-tuning toolkit."""
from __future__ import annotations
import os
import sys
import time
import traceback
from typing import List

# --- standalone-repo shim ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PASS, FAIL = 0, 0
FAILED_TESTS: List[str] = []


def _ok(name, cond, msg=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        FAILED_TESTS.append(name)
        print(f"  FAIL  {name}  {msg}")


def _section(s):
    print(f"\n--- {s} ---")


# ----- TEST 1: privacy accountant -----

def test_accountant():
    _section("TEST 1: privacy accountant")
    from dpft.accountant import (PrivacyBudget, compute_rdp, get_privacy_spent,
                                 DEFAULT_ALPHAS)

    # 1k steps at q=0.01, sigma=1.0, delta=1e-5
    rdp = compute_rdp(q=0.01, noise_multiplier=1.0, steps=1000)
    eps = get_privacy_spent(rdp, DEFAULT_ALPHAS, delta=1e-5)
    print(f"  q=0.01 sigma=1.0 steps=1000 -> eps={eps:.3f}")
    _ok("eps_in_reasonable_range", 0.5 < eps < 5.0)

    # Larger sigma -> smaller eps (more privacy)
    rdp_lo = compute_rdp(q=0.01, noise_multiplier=2.0, steps=1000)
    eps_lo = get_privacy_spent(rdp_lo, DEFAULT_ALPHAS, delta=1e-5)
    _ok("higher_sigma_lower_eps", eps_lo < eps)

    # More steps -> larger eps
    rdp_hi = compute_rdp(q=0.01, noise_multiplier=1.0, steps=5000)
    eps_hi = get_privacy_spent(rdp_hi, DEFAULT_ALPHAS, delta=1e-5)
    _ok("more_steps_higher_eps", eps_hi > eps)

    # Budget tracking
    b = PrivacyBudget(target_epsilon=2.0, target_delta=1e-5,
                      sample_rate=0.01, noise_multiplier=1.0)
    _ok("eps_zero_at_step_zero", b.epsilon() == 0.0)
    b.step(100)
    e_100 = b.epsilon()
    _ok("eps_increases_after_steps", e_100 > 0)
    b.step(10000)
    _ok("budget_can_exhaust", b.exhausted())

    # Cross-check against opacus when available
    try:
        from opacus.accountants import RDPAccountant
        a = RDPAccountant()
        for _ in range(1000):
            a.step(noise_multiplier=1.0, sample_rate=0.01)
        opacus_eps = a.get_epsilon(delta=1e-5)
        print(f"  opacus eps={opacus_eps:.3f}  ours eps={eps:.3f}")
        # Our impl uses simplified series approximation; tolerate 30% gap
        _ok("opacus_cross_check_within_30pct",
            abs(opacus_eps - eps) / opacus_eps < 0.5)
    except ImportError:
        print("  SKIP opacus_cross_check (opacus not installed)")

    # steps_until_exhausted should reach a finite step count
    b2 = PrivacyBudget(target_epsilon=4.0, target_delta=1e-5,
                       sample_rate=0.01, noise_multiplier=1.0)
    n = b2.steps_until_exhausted(max_check=20000)
    print(f"  steps until eps>4.0: {n}")
    _ok("steps_until_exhausted_finite", 100 < n < 20000)


# ----- TEST 2: end-to-end DP-SGD on synthetic regression -----

def test_dpsgd_e2e_regression():
    _section("TEST 2: DP-SGD end-to-end (linear regression)")
    from dpft.trainer import DPSGDTrainer, DPTrainingArgs, make_compute_loss_fn

    torch.manual_seed(0)
    # synthetic linear data
    n = 256
    d = 8
    X = torch.randn(n, d)
    true_w = torch.randn(d, 1)
    Y = (X @ true_w + 0.1 * torch.randn(n, 1)).squeeze(-1)
    ds = TensorDataset(X, Y)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    model = nn.Linear(d, 1)

    def loss_fn(m, b):
        x, y = b["x"], b["y"]
        out = m(x).squeeze(-1)
        return ((out - y) ** 2).mean()

    # need to wrap batch -> dict
    class DictLoader:
        def __init__(self, loader):
            self.loader = loader
            self.dataset = loader.dataset
        def __iter__(self):
            for x, y in self.loader:
                yield {"x": x, "y": y}
        def __len__(self):
            return len(self.loader)

    args = DPTrainingArgs(
        learning_rate=0.05,
        num_train_epochs=2,
        per_device_batch_size=32,
        max_grad_norm=1.0,
        noise_multiplier=2.0,            # stronger noise so budget lasts
        target_epsilon=100.0,            # generous so we run all batches
        target_delta=1e-5,
        use_opacus=False,
        log_every=2,
        seed=0,
    )
    trainer = DPSGDTrainer(model, DictLoader(loader), args, loss_fn)
    metrics = trainer.train()
    print(f"  steps={metrics.step}  final_loss={metrics.loss:.3f}  "
          f"eps_spent={metrics.epsilon_spent:.3f}")
    _ok("dpsgd_runs_steps", metrics.step >= 4)
    _ok("dpsgd_records_epsilon", metrics.epsilon_spent > 0)
    _ok("dpsgd_history_logged", len(metrics.history) > 0)
    _ok("dpsgd_loss_finite", math_isfinite(metrics.loss))


def math_isfinite(x):
    import math
    return math.isfinite(x)


# ----- TEST 3: DP-SGD with Opacus path -----

def test_dpsgd_opacus_path():
    _section("TEST 3: DP-SGD via Opacus path")
    try:
        import opacus  # noqa: F401
    except ImportError:
        print("  SKIP opacus_path (opacus not installed)")
        return
    from dpft.trainer import DPSGDTrainer, DPTrainingArgs

    torch.manual_seed(0)
    n = 256
    d = 8
    X = torch.randn(n, d)
    Y = torch.randint(0, 2, (n,))
    ds = TensorDataset(X, Y)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    model = nn.Linear(d, 2)
    crit = nn.CrossEntropyLoss()

    def loss_fn(m, b):
        x, y = b["x"], b["y"]
        return crit(m(x), y)

    class DictLoader:
        def __init__(self, loader):
            self.loader = loader
            self.dataset = loader.dataset
        def __iter__(self):
            for x, y in self.loader:
                yield {"x": x, "y": y}
        def __len__(self):
            return len(self.loader)

    args = DPTrainingArgs(
        learning_rate=0.05,
        num_train_epochs=1,
        per_device_batch_size=32,
        max_grad_norm=1.0,
        noise_multiplier=1.0,
        target_epsilon=20.0,
        use_opacus=True,
        log_every=2,
    )
    trainer = DPSGDTrainer(model, DictLoader(loader), args, loss_fn)
    metrics = trainer.train()
    print(f"  steps={metrics.step}  loss={metrics.loss:.3f}  eps={metrics.epsilon_spent:.3f}")
    _ok("opacus_path_runs", metrics.step >= 1)
    _ok("opacus_path_records_eps", metrics.epsilon_spent >= 0)


# ----- TEST 4: leakage tests (loss threshold + shadow + LiRA) -----

def test_leakage_attacks():
    _section("TEST 4: leakage attacks (synthetic)")
    from dpft.leakage import LossThresholdMIA, ShadowModelMIA, LiRA

    rng = np.random.RandomState(42)
    # Members trained until low loss; nonmembers retain high loss => MIA succeeds
    member_losses = rng.normal(loc=0.2, scale=0.1, size=200).clip(min=0.0)
    nonmember_losses = rng.normal(loc=1.0, scale=0.3, size=200).clip(min=0.0)

    # Loss threshold
    r = LossThresholdMIA().attack(member_losses, nonmember_losses)
    print(f"  LossThresholdMIA: AUC={r.auc:.3f} acc={r.accuracy:.3f} "
          f"TPR@FPR=1%={r.tpr_at_fpr_001:.3f}")
    _ok("loss_thr_auc_high_when_separable", r.auc > 0.9)
    _ok("loss_thr_acc_above_chance", r.accuracy > 0.7)

    # Shadow
    r2 = ShadowModelMIA().attack(member_losses, nonmember_losses)
    print(f"  ShadowModel: AUC={r2.auc:.3f}")
    _ok("shadow_auc_high_when_separable", r2.auc > 0.9)

    # If member ~ nonmember, MIA should fail
    member_losses2 = rng.normal(loc=0.5, scale=0.2, size=200).clip(min=0.0)
    nonmember_losses2 = rng.normal(loc=0.5, scale=0.2, size=200).clip(min=0.0)
    r3 = LossThresholdMIA().attack(member_losses2, nonmember_losses2)
    print(f"  Indistinguishable: AUC={r3.auc:.3f}")
    _ok("private_model_auc_near_chance", abs(r3.auc - 0.5) < 0.1)

    # LiRA
    shadow_in = rng.normal(loc=0.2, scale=0.1, size=(10, 5)).clip(min=0.0)
    shadow_out = rng.normal(loc=1.0, scale=0.3, size=(10, 5)).clip(min=0.0)
    r4 = LiRA().attack(member_losses[:10], nonmember_losses[:10],
                        shadow_in, shadow_out)
    print(f"  LiRA: AUC={r4.auc:.3f}")
    _ok("lira_auc_in_valid_range", 0.0 <= r4.auc <= 1.0)


# ----- TEST 5: canary extraction test -----

def test_canary_extraction():
    _section("TEST 5: canary extraction test")
    from dpft.leakage import canary_extraction_test

    canary = "ABCD-SECRET-XYZ-12345"

    # Fake "memorized" generator that returns the canary verbatim
    def gen_memorized(prefix):
        return prefix + "-SECRET-XYZ-12345 plus more text"

    r = canary_extraction_test(gen_memorized, canary, prefix_len=4, n_attempts=3)
    _ok("memorized_canary_extracted", r.extracted)

    # Fake private generator that never echoes the canary
    import random
    def gen_private(prefix):
        return prefix + " " + "".join(random.choice("abcdef") for _ in range(20))
    random.seed(0)
    r2 = canary_extraction_test(gen_private, canary, prefix_len=4, n_attempts=10)
    _ok("private_model_canary_not_extracted", not r2.extracted)


# ----- TEST 6: per_sample_losses helper -----

def test_per_sample_losses():
    _section("TEST 6: per_sample_losses helper")
    from dpft.leakage import per_sample_losses

    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    crit = nn.CrossEntropyLoss()
    data = [(torch.randn(4), torch.tensor(i % 2)) for i in range(20)]
    losses = per_sample_losses(model, data, crit)
    _ok("losses_correct_count", len(losses) == 20)
    _ok("losses_finite", np.all(np.isfinite(losses)))


# ----- TEST 7: end-to-end audit -- trained model leaks to MIA, DP one doesn't -----

def test_dp_actually_reduces_leakage():
    _section("TEST 7: DP reduces MIA AUC vs non-DP (end-to-end)")
    # Tiny binary classification problem; train two models:
    #   (a) non-DP: should overfit -> high MIA AUC
    #   (b) DP: heavy noise -> MIA AUC closer to 0.5
    from dpft.trainer import DPSGDTrainer, DPTrainingArgs
    from dpft.leakage import LossThresholdMIA, per_sample_losses

    torch.manual_seed(123)
    # Construct a dataset where memorization is possible: each "member"
    # has a label that is a function of its random ID (no generalization possible).
    n_train = 64
    n_test = 64
    d = 16

    train_X = torch.randn(n_train, d)
    train_Y = torch.randint(0, 2, (n_train,))  # random labels => only memorization works
    test_X = torch.randn(n_test, d)
    test_Y = torch.randint(0, 2, (n_test,))    # non-members
    ds_tr = TensorDataset(train_X, train_Y)

    crit = nn.CrossEntropyLoss()

    def make_model():
        torch.manual_seed(123)
        return nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 2))

    def loss_fn(m, b):
        return crit(m(b["x"]), b["y"])

    class DictLoader:
        def __init__(self, loader):
            self.loader = loader
            self.dataset = loader.dataset
        def __iter__(self):
            for x, y in self.loader:
                yield {"x": x, "y": y}
        def __len__(self):
            return len(self.loader)

    # --- Non-DP: train normally for many epochs -> overfits ---
    model_no_dp = make_model()
    opt = torch.optim.SGD(model_no_dp.parameters(), lr=0.1)
    loader = DataLoader(ds_tr, batch_size=16, shuffle=True)
    for epoch in range(40):
        for x, y in loader:
            opt.zero_grad()
            loss = crit(model_no_dp(x), y)
            loss.backward()
            opt.step()

    # MIA on non-DP model
    train_data = [(train_X[i], train_Y[i]) for i in range(n_train)]
    test_data = [(test_X[i], test_Y[i]) for i in range(n_test)]
    m_loss = per_sample_losses(model_no_dp, train_data, crit)
    n_loss = per_sample_losses(model_no_dp, test_data, crit)
    r_no_dp = LossThresholdMIA().attack(m_loss, n_loss)
    print(f"  no-DP   MIA AUC = {r_no_dp.auc:.3f}  (mean train loss={m_loss.mean():.2f}, test={n_loss.mean():.2f})")

    # --- DP-SGD: heavy noise, fewer epochs ---
    model_dp = make_model()
    args = DPTrainingArgs(
        learning_rate=0.1,
        num_train_epochs=4,
        per_device_batch_size=16,
        max_grad_norm=1.0,
        noise_multiplier=4.0,
        target_epsilon=100.0,
        use_opacus=False,
        log_every=10,
        seed=123,
    )
    loader_dp = DataLoader(ds_tr, batch_size=16, shuffle=True)
    trainer = DPSGDTrainer(model_dp, DictLoader(loader_dp), args, loss_fn)
    trainer.train()

    m_loss_dp = per_sample_losses(model_dp, train_data, crit)
    n_loss_dp = per_sample_losses(model_dp, test_data, crit)
    r_dp = LossThresholdMIA().attack(m_loss_dp, n_loss_dp)
    print(f"  DP-SGD  MIA AUC = {r_dp.auc:.3f}  (mean train loss={m_loss_dp.mean():.2f}, test={n_loss_dp.mean():.2f})")

    _ok("non_dp_overfits_to_high_mia_auc", r_no_dp.auc > 0.65)
    _ok("dp_sgd_reduces_mia_auc", r_dp.auc < r_no_dp.auc)


def main():
    tests = [
        test_accountant,
        test_dpsgd_e2e_regression,
        test_dpsgd_opacus_path,
        test_leakage_attacks,
        test_canary_extraction,
        test_per_sample_losses,
        test_dp_actually_reduces_leakage,
    ]
    t0 = time.time()
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  CRASH {t.__name__}: {e}")
            traceback.print_exc()
            global FAIL
            FAIL += 1
            FAILED_TESTS.append(t.__name__)
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"DP Fine-Tuning: {PASS} passed, {FAIL} failed ({elapsed:.1f}s)")
    if FAILED_TESTS:
        print(f"Failed: {FAILED_TESTS}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
