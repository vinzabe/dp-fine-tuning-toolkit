"""CLI: train with DP-SGD and report the privacy budget; run the empirical audit.

Exit codes: 0 ok / audit consistent, 2 audit VIOLATION (empirical eps exceeds
theoretical), 1 error.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .accountant import compute_epsilon, noise_for_epsilon
from .audit import run_audit
from .dpsgd import DPSGDConfig, train
from .synth import make_classification

EXIT_OK, EXIT_ERROR, EXIT_VIOLATION = 0, 1, 2


def cmd_train(a: argparse.Namespace) -> int:
    X, y = make_classification(a.n, seed=a.seed)
    cfg = DPSGDConfig(clip_norm=a.clip, noise_multiplier=a.noise,
                      lot_size=a.lot, epochs=a.epochs, seed=a.seed)
    model = train(X, y, cfg)
    spent = compute_epsilon(noise_multiplier=a.noise,
                            sampling_rate=model.sampling_rate,
                            steps=model.steps, delta=a.delta)
    acc = float(((model.predict_proba(X) > 0.5).astype(int) == y).mean())
    out = {"epsilon": spent.epsilon, "delta": spent.delta, "steps": model.steps,
           "noise_multiplier": a.noise, "clip_norm": a.clip,
           "train_accuracy": acc}
    if a.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"DP-SGD trained: (ε={spent.epsilon:.3f}, δ={spent.delta:.0e}) "
              f"over {model.steps} steps")
        print(f"  noise σ={a.noise}  clip C={a.clip}  train acc={acc:.1%}")
    return EXIT_OK


def cmd_audit(a: argparse.Namespace) -> int:
    X, y = make_classification(a.n, seed=a.seed)
    cfg = DPSGDConfig(clip_norm=a.clip, noise_multiplier=a.noise,
                      lot_size=a.lot, epochs=a.epochs, seed=a.seed)
    steps = max(1, (a.n // a.lot)) * a.epochs
    theoretical = compute_epsilon(noise_multiplier=a.noise,
                                  sampling_rate=min(1.0, a.lot / a.n),
                                  steps=steps, delta=a.delta).epsilon
    result = run_audit(X, y, cfg, theoretical_epsilon=theoretical,
                       delta=a.delta, trials=a.trials, seed=a.seed)
    if a.json:
        print(json.dumps({
            "theoretical_epsilon": result.theoretical_epsilon,
            "empirical_epsilon": result.empirical_epsilon,
            "attack_tpr": result.attack_tpr, "attack_fpr": result.attack_fpr,
            "consistent": result.consistent}, indent=2))
    else:
        print(f"Privacy audit ({result.trials} trials):")
        print(f"  attack TPR={result.attack_tpr:.2f}  FPR={result.attack_fpr:.2f}")
        print(f"  {result.note}")
    return EXIT_OK if result.consistent else EXIT_VIOLATION


def cmd_budget(a: argparse.Namespace) -> int:
    noise = noise_for_epsilon(a.target_epsilon, sampling_rate=a.q,
                              steps=a.steps, delta=a.delta)
    print(f"noise_multiplier ≈ {noise:.4f} achieves ε={a.target_epsilon} "
          f"(δ={a.delta:.0e}) over {a.steps} steps at q={a.q}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dpft", description=__doc__)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--n", type=int, default=400)
        sp.add_argument("--clip", type=float, default=1.0)
        sp.add_argument("--noise", type=float, default=1.0)
        sp.add_argument("--lot", type=int, default=32)
        sp.add_argument("--epochs", type=int, default=20)
        sp.add_argument("--delta", type=float, default=1e-5)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--json", action="store_true")

    t = sub.add_parser("train", help="train with DP-SGD and report the budget")
    common(t)
    t.set_defaults(func=cmd_train)

    au = sub.add_parser("audit", help="empirically audit the privacy guarantee")
    common(au)
    au.add_argument("--trials", type=int, default=200)
    au.set_defaults(func=cmd_audit)

    b = sub.add_parser("budget", help="noise multiplier for a target epsilon")
    b.add_argument("target_epsilon", type=float)
    b.add_argument("--q", type=float, default=0.01)
    b.add_argument("--steps", type=int, default=1000)
    b.add_argument("--delta", type=float, default=1e-5)
    b.set_defaults(func=cmd_budget)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rc: int = args.func(args)
        return rc
    except (ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
