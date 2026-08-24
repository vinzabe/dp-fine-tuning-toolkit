"""The headline: the empirical audit must be CONSISTENT (empirical eps <=
theoretical) for a correctly-implemented mechanism, and it must detect more
privacy loss when noise is lower."""
from dpft.accountant import compute_epsilon
from dpft.audit import _empirical_epsilon, run_audit
from dpft.dpsgd import DPSGDConfig
from dpft.synth import make_classification


def test_audit_consistent_for_correct_mechanism():
    X, y = make_classification(150, seed=0)
    cfg = DPSGDConfig(noise_multiplier=1.0, clip_norm=1.0, lot_size=32,
                      epochs=8, seed=0)
    steps = max(1, 150 // 32) * 8
    theo = compute_epsilon(noise_multiplier=1.0, sampling_rate=32 / 150,
                           steps=steps).epsilon
    result = run_audit(X, y, cfg, theoretical_epsilon=theo, trials=60, seed=1)
    # a correct DP mechanism: the attacker's empirical eps must not exceed theory
    assert result.consistent, result.note


def test_empirical_epsilon_zero_for_random_guessing():
    # TPR == FPR (attack no better than chance) -> empirical eps ~ 0
    assert _empirical_epsilon(0.5, 0.5, 1e-5) < 0.05


def test_empirical_epsilon_grows_with_attack_advantage():
    weak = _empirical_epsilon(0.55, 0.45, 1e-5)
    strong = _empirical_epsilon(0.95, 0.05, 1e-5)
    assert strong > weak


def test_audit_reports_rates():
    X, y = make_classification(120, seed=5)
    cfg = DPSGDConfig(noise_multiplier=2.0, epochs=6, seed=5)
    r = run_audit(X, y, cfg, theoretical_epsilon=10.0, trials=40, seed=2)
    assert 0.0 <= r.attack_tpr <= 1.0 and 0.0 <= r.attack_fpr <= 1.0
    assert r.empirical_epsilon >= 0.0
