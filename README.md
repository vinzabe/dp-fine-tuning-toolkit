# dp-fine-tuning-toolkit

**Differentially-private training whose privacy claim is *audited*, not just asserted.**

Differential privacy has a credibility problem. Papers report an ε from an accountant, but accounting bugs are common, and a wrong clip or a miscomposed step silently inflates the real privacy loss. An ε nobody can check is an ε nobody should trust.

This toolkit ties three things together so the number means something:

1. **DP-SGD** — per-example gradient clipping (bounds one record's influence) + Gaussian noise (the mechanism).
2. **An RDP accountant** for the subsampled Gaussian mechanism → (ε, δ).
3. **An empirical privacy audit** — a membership-inference attack that produces a statistical **lower bound** on ε. If the empirical lower bound ever exceeds the theoretical ε, the guarantee is violated and the audit fails loudly.

```
$ dpft audit --n 150 --noise 1.0 --trials 60
Privacy audit (60 trials):
  attack TPR=0.58  FPR=0.44
  empirical eps lower bound 0.31 <= theoretical 3.42: consistent
```

That last line is the whole point. Not "we computed ε=3.42" but "we computed 3.42 **and** a real membership-inference attacker could not demonstrate more than 0.31, which is consistent with the claim."

## Quickstart (60 seconds)

```bash
git clone https://github.com/vinzabe/dp-fine-tuning-toolkit && cd dp-fine-tuning-toolkit
python -m pip install -e ".[dev]"

dpft train --noise 1.0                 # train + report the (ε, δ) budget
dpft budget 2.0 --q 0.01 --steps 1000  # what noise achieves a target ε?
dpft audit --noise 1.0 --trials 100    # empirically validate the guarantee
```

Exit codes: `0` audit consistent, `2` **audit violation** (empirical ε exceeds theoretical), `1` error.

## The privacy/utility trade-off, shown not told

More noise buys more privacy at the cost of accuracy — and the toolkit makes that concrete and testable:

| noise σ | ε (q=0.01, 1000 steps) | learns? |
|---|---|---|
| 0.5 | ~2.9 | well |
| 1.0 | ~1.5 | ok |
| 2.0 | ~0.7 | weakly |
| 4.0 | ~0.3 | barely |

`test_dpsgd.py::test_less_noise_learns_better` asserts the low-noise model out-accuracies the high-noise one — the trade-off, encoded as a test.

## The accountant (standard math, no black box)

RDP for the subsampled Gaussian mechanism (Mironov 2017; Wang–Balle–Kasiviswanathan subsampling), composed over steps, converted to (ε, δ) over a search of Rényi orders. Implemented in ~110 lines of numpy so the math is auditable. Its invariants are tested: ε **decreases** with noise, **increases** with steps and sampling rate, is 0 at 0 steps, and `noise_for_epsilon` round-trips.

## How the audit works

For many trials: flip a coin for whether a strong "canary" record is in the training set, train a model, and let a membership-inference attack guess inclusion from the canary's loss. Aggregate the attack's (TPR, FPR) and convert to an empirical ε lower bound via the DP hypothesis-testing characterization. A correct mechanism keeps the attacker at or below the theoretical ε; `test_audit.py` asserts consistency for the shipped mechanism.

Auditing is stronger than accounting because it catches implementation bugs an accountant is blind to — a clip applied after summing, a noise scale off by C, a step counted wrong. `docs/adr/0002` covers why the audit is the headline.

## Development

```bash
python -m pip install -e ".[dev]"
pytest --cov=dpft       # 19 tests, incl. accountant invariants + audit consistency
mypy --strict src/dpft  # clean (targets 3.12 for numpy stubs; runtime 3.11+)
ruff check src tests    # clean
```

## License

MIT © vinzabe
