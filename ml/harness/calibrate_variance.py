"""Phase 4: fit the day-factor sigma, then the scoring calibration constant.

    ml/.venv/Scripts/python -m ml.harness.calibrate_variance

Fitted at a NEUTRAL role mix on purpose. The harness picks roles by rolling dice
against a phase mix, and that randomness is an artifact -- real players choose
deliberately, they don't roll. Fitting the day factor against it would credit
conditions variance for spread that comes from the harness's own coin flips.

Order matters: sigma first (it barely moves the mean), then the scoring constant.
"""

from __future__ import annotations

import argparse
import json
import os

from ml.harness.calibrate import bisect
from ml.harness.run_baseline import ARTIFACTS, real_reference
from ml.harness.run_model import model_ball_fn, model_extras_fn
from ml.harness.simulate import RoleMix, run_batch
from ml.harness.stats import summarize
from ml.runtime.model import OutcomeModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--since", type=int, default=2023,
                    help="calibrate against seasons >= this (default 2023: the game "
                         "as currently played). Pass 0 for the all-time average.")
    args = ap.parse_args()

    real, plans = real_reference(season_min=args.since or None)
    print(f"reference: {real['n_innings']} innings"
          f"{f', seasons >= {args.since}' if args.since else ' (all-time)'}")
    model = OutcomeModel.load()
    extras = model_extras_fn(model)
    mix = RoleMix.neutral()

    def run(sigma, calib, out_calib):
        sims = run_batch(plans,
                         model_ball_fn(model, calibration=calib,
                                       out_calibration=out_calib),
                         n=args.n, seed=args.seed, role_mix=mix,
                         extras_fn=extras, day_sigma=sigma, progress_every=0)
        return summarize(sims)

    print(f"targets: innings SD {real['innings_sd']:.2f}, "
          f"mean {real['innings_mean']:.1f}, all-out {real['allout_rate']:.2f}%\n")

    # Three knobs, cycled twice because they interact: scaling scoring up pushes
    # Out down after renormalisation, and cutting wickets lengthens innings, which
    # moves the mean back. Two passes is enough to settle.
    sigma, calib, out_calib = 0.26, 1.0, 1.0
    for rnd in range(args.rounds):
        print(f"[round {rnd + 1}/{args.rounds}] day_sigma -> innings SD")
        sigma, _ = bisect(lambda s: run(s, calib, out_calib)["innings_sd"],
                          real["innings_sd"], 0.0, 0.60, tol=0.7, label="sigma    ")

        print(f"\n[round {rnd + 1}/{args.rounds}] out_calibration -> all-out rate")
        out_calib, _ = bisect(lambda o: run(sigma, calib, o)["allout_rate"],
                              real["allout_rate"], 0.60, 2.20, tol=1.2, label="out_calib")

        print(f"\n[round {rnd + 1}/{args.rounds}] calibration -> innings mean")
        calib, _ = bisect(lambda c: run(sigma, c, out_calib)["innings_mean"],
                          real["innings_mean"], 0.90, 1.60, tol=1.5, label="calib    ")
        print()

    print(f"\nfitted: day_sigma={sigma:.4f}  calibration={calib:.4f}  "
          f"out_calibration={out_calib:.4f}")

    final = run(sigma, calib, out_calib)
    print("\nat the fitted values (neutral roles):")
    for k in ("innings_mean", "innings_sd", "allout_rate", "pct_over_200",
              "pct_under_120", "dot_pct", "out_pct", "wkts_per_innings",
              "over_rr_autocorr"):
        print(f"  {k:<20} real {real[k]:>7.2f}   model {final[k]:>7.2f}")

    os.makedirs(ARTIFACTS, exist_ok=True)
    path = os.path.join(ARTIFACTS, "model_calibration.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"day_sigma": sigma, "calibration": calib,
                   "out_calibration": out_calib,
                   "fitted_at": "neutral", "n": args.n}, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
