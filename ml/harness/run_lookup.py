"""Phase 1 quick win: measured (over x outcome) base vs the classic engine.

    python -m ml.harness.run_lookup [--n 10000]

No model. The engine's single global BASELINE_WEIGHTS and its hand-picked
PHASE_EFFECTS are replaced by the real league-average distribution for each over,
straight from the ball table; extras get real per-over rates instead of a flat 4%
at a 70/30 split. Everything else -- the Stage 1/2/3 player ratios, the gambits,
the roles, the cascade -- is the classic engine, untouched.

The point is to prove the ETL and the harness agree before betting anything on a
model, and to see how much of the gap closes without one.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from src.engine.simulator import calculate_single_ball
from ml.harness import compare
from ml.harness.run_baseline import ARTIFACTS, real_reference
from ml.harness.simulate import RoleMix, run_batch
from ml.harness.stats import summarize
from ml.runtime import lookup
from ml.runtime.adapter import make_ball_fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--neutral", action="store_true")
    ap.add_argument("--calibration", type=float, default=1.0,
                    help="global multiplier on scoring buckets (absorbs role inflation)")
    ap.add_argument("--out-calibration", type=float, default=1.0,
                    help="global multiplier on Out")
    ap.add_argument("--no-stages", action="store_true",
                    help="skip Stage 1/2/3 player ratios -- isolates how much the "
                         "stages (rather than the base) are responsible for the gap")
    ap.add_argument("--no-roles", action="store_true",
                    help="also skip Stage 4/5 role bonuses -- leaves base + cascade only")
    ap.add_argument("--no-classic", action="store_true", help="skip the classic run")
    args = ap.parse_args()

    os.makedirs(ARTIFACTS, exist_ok=True)
    mix = RoleMix.neutral() if args.neutral else RoleMix.realistic()
    label = "neutral" if args.neutral else "realistic"
    if args.no_roles:
        label += "+noroles"
    if args.no_stages:
        label += "+nostages"

    print("[1/3] replaying real matches ...", flush=True)
    real, plans = real_reference()
    print(f"      {real['n_innings']} real innings")

    classic = None
    if not args.no_classic:
        print(f"\n[2/3] classic engine, {args.n} innings ({label}) ...", flush=True)
        t0 = time.time()
        classic = summarize(run_batch(plans, calculate_single_ball, n=args.n,
                                      seed=args.seed, role_mix=mix, progress_every=0))
        print(f"      {time.time() - t0:.1f}s")

    print(f"\n[3/3] measured-lookup base, {args.n} innings ({label}, "
          f"calibration {args.calibration:g}) ...", flush=True)
    base_by_over, extras_by_over = lookup.measure()
    ball_fn = make_ball_fn(
        lookup.make_provider(base_by_over),
        # the base is a league average, so the stages supply per-player
        # differentiation -- unless we're isolating their contribution
        player_stages=not args.no_stages,
        calibration=args.calibration,
        out_calibration=args.out_calibration,
    )
    t0 = time.time()
    sims = run_batch(plans, ball_fn, n=args.n, seed=args.seed, role_mix=mix,
                     extras_fn=lookup.make_extras_fn(extras_by_over),
                     use_roles=not args.no_roles, progress_every=0)
    print(f"      {time.time() - t0:.1f}s")
    sim = summarize(sims)

    with open(os.path.join(ARTIFACTS, f"lookup_{label}.json"), "w", encoding="utf-8") as fh:
        json.dump(sim, fh, indent=2)

    if classic:
        print("\n=== CLASSIC ENGINE ===")
        n_classic = compare.report(real, classic, title="classic")
    print("=== MEASURED LOOKUP ===")
    n_lookup = compare.report(real, sim, title="lookup")

    if classic:
        print(f"  classic: {n_classic}/16 metrics off   ->   "
              f"lookup: {n_lookup}/16 metrics off\n")


if __name__ == "__main__":
    main()
