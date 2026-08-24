"""The accountant is DP's trust anchor; its core properties must hold."""
import pytest

from dpft.accountant import compute_epsilon, noise_for_epsilon


def test_more_noise_less_epsilon():
    eps = [compute_epsilon(noise_multiplier=s, sampling_rate=0.01, steps=1000).epsilon
           for s in (0.5, 1.0, 2.0, 4.0)]
    assert eps == sorted(eps, reverse=True)   # strictly decreasing in noise


def test_more_steps_more_epsilon():
    e1 = compute_epsilon(noise_multiplier=1.0, sampling_rate=0.01, steps=100).epsilon
    e2 = compute_epsilon(noise_multiplier=1.0, sampling_rate=0.01, steps=1000).epsilon
    assert e2 > e1


def test_higher_sampling_more_epsilon():
    lo = compute_epsilon(noise_multiplier=1.0, sampling_rate=0.001, steps=500).epsilon
    hi = compute_epsilon(noise_multiplier=1.0, sampling_rate=0.05, steps=500).epsilon
    assert hi > lo


def test_zero_steps_zero_epsilon():
    assert compute_epsilon(noise_multiplier=1.0, sampling_rate=0.01, steps=0).epsilon == 0.0


def test_noise_for_epsilon_roundtrips():
    target = 2.0
    sigma = noise_for_epsilon(target, sampling_rate=0.01, steps=1000)
    got = compute_epsilon(noise_multiplier=sigma, sampling_rate=0.01, steps=1000).epsilon
    assert abs(got - target) < 0.15


@pytest.mark.parametrize("kw", [
    {"noise_multiplier": 0}, {"noise_multiplier": -1},
    {"sampling_rate": 0}, {"sampling_rate": 1.5}, {"steps": -1}])
def test_invalid_args_rejected(kw):
    base = {"noise_multiplier": 1.0, "sampling_rate": 0.01, "steps": 100}
    base.update(kw)
    with pytest.raises(ValueError):
        compute_epsilon(**base)
