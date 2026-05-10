"""CLI for the DP fine-tuning toolkit.

Subcommands:
  budget    -- compute privacy budget given training schedule
  steps     -- compute max #steps allowable for a target epsilon
  audit     -- run MIA leakage tests on a trained model checkpoint
"""
from __future__ import annotations
import argparse
import json
import os
import sys

# --- standalone-repo shim ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
sys.path.insert(0, _PROJECT_ROOT)

from .accountant import PrivacyBudget, compute_rdp, get_privacy_spent, DEFAULT_ALPHAS


def cmd_budget(args):
    budget = PrivacyBudget(
        target_epsilon=args.target_epsilon,
        target_delta=args.delta,
        sample_rate=args.sample_rate,
        noise_multiplier=args.sigma,
        steps=args.steps,
    )
    eps = budget.epsilon()
    print(json.dumps({
        "sample_rate": args.sample_rate,
        "noise_multiplier": args.sigma,
        "delta": args.delta,
        "steps": args.steps,
        "epsilon_spent": eps,
        "target_epsilon": args.target_epsilon,
        "exhausted": eps > args.target_epsilon,
    }, indent=2))


def cmd_steps(args):
    budget = PrivacyBudget(
        target_epsilon=args.target_epsilon,
        target_delta=args.delta,
        sample_rate=args.sample_rate,
        noise_multiplier=args.sigma,
    )
    steps = budget.steps_until_exhausted(max_check=args.max_check)
    print(json.dumps({
        "sample_rate": args.sample_rate,
        "noise_multiplier": args.sigma,
        "delta": args.delta,
        "target_epsilon": args.target_epsilon,
        "max_steps": steps,
    }, indent=2))


def cmd_audit(args):
    """Stub: real audit needs a model + dataset; CLI offers smoke check.

    For full leakage testing, use dpft.leakage.{LossThresholdMIA, ShadowModelMIA, LiRA}
    programmatically with your actual model and member/nonmember splits.
    """
    print("dpft audit: provide a trained model + member/nonmember splits "
          "via the Python API (see examples/leakage_demo.py).")


def main(argv=None):
    p = argparse.ArgumentParser(prog="dpft")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("budget", help="compute eps for a given training plan")
    b.add_argument("--sample-rate", type=float, required=True)
    b.add_argument("--sigma", type=float, required=True)
    b.add_argument("--steps", type=int, required=True)
    b.add_argument("--delta", type=float, default=1e-5)
    b.add_argument("--target-epsilon", type=float, default=8.0)
    b.set_defaults(func=cmd_budget)

    s = sub.add_parser("steps", help="compute max steps for a target eps")
    s.add_argument("--sample-rate", type=float, required=True)
    s.add_argument("--sigma", type=float, required=True)
    s.add_argument("--target-epsilon", type=float, required=True)
    s.add_argument("--delta", type=float, default=1e-5)
    s.add_argument("--max-check", type=int, default=200000)
    s.set_defaults(func=cmd_steps)

    a = sub.add_parser("audit", help="leakage test stub")
    a.set_defaults(func=cmd_audit)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
