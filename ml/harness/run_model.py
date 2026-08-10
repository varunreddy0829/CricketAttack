"""Phase 3 acceptance test: the learned model against the classic engine.

    ml/.venv/Scripts/python -m ml.harness.run_model [--n 6000] [--day-sigma 0.25]

Both paths run the same harness over the same real XIs and bowling plans, with the
same role mix. The classic engine keeps all six of its stages; the model path
replaces Stages 0-3 and the pitch/phase half of 3.5, and keeps the gambits and the
role bonuses exactly as they are.

The wicket cascade (Stage 6) is OFF by default for the model, unlike the classic
engine. It was tuned to make consecutive-wicket overs rarer than an iid model would
produce, back when the engine had no way to know a new batter had just arrived. The
model already sees that directly (striker_balls, is_set, partnership_balls all read
0 on a fresh batter's first ball) and learned the real effect from history, so
layering the old hand-tuned damping on top double-suppresses it. Measured: WITH the
cascade the model under-shoots all-out rate (8.34% vs a real 11.32%), wickets/innings
(5.65 vs 6.09), and even three-wicket-overs itself (0.04% vs a real 0.20%, i.e. now
too RARE); WITHOUT it every one of those lands closer to reality. Pass --cascade to
restore it for a side-by-side comparison.
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
from ml.runtime.adapter import make_ball_fn
from ml.runtime.longevity import LONGEVITY_DIAL, scores_for
from ml.runtime.model import OutcomeModel


def model_ball_fn(model: OutcomeModel, *, calibration=1.0, out_calibration=1.0,
                  cascade=False, era_id=None, longevity_dial=None, shrink=0.0,
                  target='anchor'):
    """The harness must build the SAME ball function the game plays with.

    The longevity layer moves Out, and Out compounds over 120 balls, so a gate
    run without it would be certifying an engine nobody plays. `longevity_dial`
    defaults to the live constant; pass 0.0 to measure the layer's own cost.
    """
    if shrink:
        from ml.runtime.players import load_players
        model.shrink_target = target
        model.set_shrinkage(list(load_players(era_id).values()), shrink)
    scores = None
    dial = LONGEVITY_DIAL if longevity_dial is None else longevity_dial
    if dial > 0.0:
        try:
            scores = scores_for(era_id)
        except Exception:
            dial = 0.0
    return make_ball_fn(
        model.base_provider(),
        player_stages=False,       # the model already knows who is batting
        cascade=cascade,
        calibration=calibration,
        out_calibration=out_calibration,
        longevity_scores=scores,
        longevity_dial=dial,
    )


def model_extras_fn(model: OutcomeModel):
    """Genuinely from the model this time -- `predict()` already computes wide/
    no-ball as 2 of its 9 outputs, using the full situation AND the specific
    bowler (wides are heavily bowler-dependent: a wild spinner and a metronomic
    quick don't share a rate). Earlier this called the flat per-over lookup table
    instead and threw the model's own prediction away; that was leftover wiring
    from before the model existed, not a deliberate choice.
    """
    def extras_fn(striker, bowler, ctx):
        _, p_wide, p_nb = model.predict(striker, bowler, ctx)
        return p_wide, p_nb
    return extras_fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--neutral", action="store_true")
    ap.add_argument("--calibration", type=float, default=1.0)
    ap.add_argument("--out-calibration", type=float, default=1.0)
    ap.add_argument("--day-sigma", type=float, default=0.0)
    ap.add_argument("--cascade", action="store_true",
                    help="restore the wicket cascade for the model path (off by "
                         "default -- see the module docstring for why)")
    ap.add_argument("--target", default="anchor",
                    choices=("anchor", "mean", "zero", "replacement"))
    ap.add_argument("--shrink", type=float, default=0.0,
                    help="regress player effects by balls/(balls+K); 0 = off")
    ap.add_argument("--longevity-dial", type=float, default=None,
                    help="override the live longevity dial; 0 measures its cost")
    ap.add_argument("--no-classic", action="store_true")
    ap.add_argument("--since", type=int, default=2023,
                    help="score against seasons >= this (default 2023). 0 = all-time.")
    ap.add_argument("--era", default=None,
                    help="score an era model against that era's own real innings")
    args = ap.parse_args()

    era = None
    if args.era:
        from ml.etl import eras as E
        era = E.get(args.era)

    mix = RoleMix.neutral() if args.neutral else RoleMix.realistic()
    label = "neutral" if args.neutral else "realistic"

    print("[1/3] replaying real matches ...", flush=True)
    real, plans = real_reference(season_min=None if era else (args.since or None),
                                 era=era)
    window = (f"{era.first}-{era.last}" if era else
              (f"seasons >= {args.since}" if args.since else "all-time"))
    print(f"      {real['n_innings']} real innings ({window})")

    classic = None
    if not args.no_classic:
        print(f"\n[2/3] classic engine, {args.n} innings ...", flush=True)
        t0 = time.time()
        classic = summarize(run_batch(plans, calculate_single_ball, n=args.n,
                                      seed=args.seed, role_mix=mix, progress_every=0))
        print(f"      {time.time() - t0:.1f}s")

    print(f"\n[3/3] learned model, {args.n} innings "
          f"(calib {args.calibration:g}/{args.out_calibration:g}, "
          f"day sigma {args.day_sigma:g}) ...", flush=True)
    model = OutcomeModel.load(era_id=era.id if era else None)
    t0 = time.time()
    sims = run_batch(
        plans,
        model_ball_fn(model, calibration=args.calibration,
                      out_calibration=args.out_calibration,
                      cascade=args.cascade, era_id=era.id if era else None,
                      longevity_dial=args.longevity_dial,
                      shrink=args.shrink, target=args.target),
        n=args.n, seed=args.seed, role_mix=mix,
        extras_fn=model_extras_fn(model),
        day_sigma=args.day_sigma, era_id=era.id if era else None,
        progress_every=0,
    )
    print(f"      {time.time() - t0:.1f}s")
    sim = summarize(sims)

    os.makedirs(ARTIFACTS, exist_ok=True)
    with open(os.path.join(ARTIFACTS, f"model_{label}.json"), "w", encoding="utf-8") as fh:
        json.dump(sim, fh, indent=2)

    if classic:
        print("\n=== CLASSIC ENGINE ===")
        n_classic = compare.report(real, classic, title="classic")
    print("=== LEARNED MODEL ===")
    n_model = compare.report(real, sim, title="model")
    if classic:
        print(f"  classic: {n_classic}/16 off   ->   model: {n_model}/16 off\n")


if __name__ == "__main__":
    main()
