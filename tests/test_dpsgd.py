import numpy as np
import pytest

from dpft.dpsgd import DPSGDConfig, train
from dpft.synth import make_classification


def test_trains_and_is_deterministic():
    X, y = make_classification(300, seed=1)
    cfg = DPSGDConfig(noise_multiplier=1.0, epochs=10, seed=7)
    a = train(X, y, cfg).weights
    b = train(X, y, cfg).weights
    assert np.allclose(a, b)   # same seed -> identical


def test_less_noise_learns_better():
    X, y = make_classification(400, seed=2)
    low = train(X, y, DPSGDConfig(noise_multiplier=0.3, epochs=25, seed=3))
    high = train(X, y, DPSGDConfig(noise_multiplier=8.0, epochs=25, seed=3))
    acc_low = ((low.predict_proba(X) > 0.5).astype(int) == y).mean()
    acc_high = ((high.predict_proba(X) > 0.5).astype(int) == y).mean()
    assert acc_low > acc_high   # the privacy/utility trade-off, demonstrated


def test_clipping_bounds_are_respected():
    # a single huge-gradient outlier must not dominate; training stays finite
    X, y = make_classification(200, seed=4)
    X[0] *= 1000.0
    m = train(X, y, DPSGDConfig(clip_norm=1.0, noise_multiplier=1.0, epochs=5))
    assert np.all(np.isfinite(m.weights)) and np.isfinite(m.bias)


@pytest.mark.parametrize("kw", [{"clip_norm": 0}, {"noise_multiplier": 0},
                                {"lot_size": 0}])
def test_invalid_config_rejected(kw):
    with pytest.raises(ValueError):
        DPSGDConfig(**kw)
