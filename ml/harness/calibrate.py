"""Fit the two global calibration constants against the harness.

    python -m ml.harness.calibrate [--n 3000]

Two knobs, fitted by bisection, in this order:

  out_calibration  matched on ALL-OUT RATE, not on per-ball Out%. The wicket rate
                   compounds across 120 balls -- the classic engine is only 1.4x
                   too high per ball but 6.8x too high on all-out rate -- so the
                   innings-level statistic is the one that has to be right.

  calibration      matched on innings mean, fitted second because fixing the wicket
                   rate lengthens innings and moves the mean on its own.

One pass of each is usually enough; the two interact weakly and the loop repeats
until both are inside tolerance.
"""

from __future__ import annotations

import argparse
import json
import os

from ml.harness.run_baseline import ARTIFACTS, real_reference
from ml.harness.simulate import RoleMix, run_batch
from ml.harness.stats import summarize
from ml.runtime import lookup
from ml.runtime.adapter import make_ball_fn


def _measure(plans, *, n, seed, mix, calib, out_calib, base_by_over, extras_fn,
             player_stages=True):
    ball_fn = make_ball_fn(
        lookup.make_provider(base_by_over),
        player_stages=player_stages,
        calibration=calib,
        out_calibration=out_calib,
    )
    sims = run_batch(plans, ball_fn, n=n, seed=seed, role_mix=mix,
                     extras_fn=extras_fn, progress_every=0)
    return summarize(sims)


def bisect(fn, target, lo, hi, *, tol, max_iter=8, label=""):
    """Find x in [lo, hi] with fn(x) ~ target. `fn` must be monotone increasing."""
    for i in range(max_iter):
        mid = 0.5 * (lo + hi)
        got = fn(mid)
        print(f"    {label} {mid:.4f} -> {got:8.2f}  (target {target:.2f})", flush=True)
        if abs(got - target) <= tol:
            return mid, got
        if got < target:
            lo = mid
        else:
            hi = mid
    return mid, got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="innings per evaluation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    real, plans = real_reference()
    mix = RoleMix.realistic()
    base_by_over, extras_by_over = lookup.measure()
    extras_fn = lookup.make_extras_fn(extras_by_over)

    target_allout = real["allout_rate"]
    target_mean = real["innings_mean"]
    print(f"targets: all-out {target_allout:.2f}%, innings mean {target_mean:.1f}\n")

    calib, out_calib = 1.0, 1.0

    for rnd in range(args.rounds):
        print(f"[round {rnd + 1}] out_calibration -> all-out rate")
        out_calib, _ = bisect(
            lambda x: _measure(plans, n=args.n, seed=args.seed, mix=mix, calib=calib,
                               out_calib=x, base_by_over=base_by_over,
                               extras_fn=extras_fn)["allout_rate"],
            target_allout, 0.20, 1.20, tol=1.5, label="out_calib",
        )

        print(f"\n[round {rnd + 1}] calibration -> innings mean")
        calib, _ = bisect(
            lambda x: _measure(plans, n=args.n, seed=args.seed, mix=mix, calib=x,
                               out_calib=out_calib, base_by_over=base_by_over,
                               extras_fn=extras_fn)["innings_mean"],
            target_mean, 0.70, 1.30, tol=2.0, label="calib    ",
        )
        print()

    print(f"\nfitted: calibration={calib:.4f}  out_calibration={out_calib:.4f}")
    os.makedirs(ARTIFACTS, exist_ok=True)
    path = os.path.join(ARTIFACTS, "calibration.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"calibration": calib, "out_calibration": out_calib,
                   "fitted_on": "lookup", "n": args.n}, fh, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
