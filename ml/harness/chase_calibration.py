"""The strongest test the plan calls for: does the model get CHASES right?

    ml/.venv/Scripts/python -m ml.harness.chase_calibration [--matches 400] [--sims 40]

Every other check so far scores full innings against distributional targets. This
is the sharper test: take a REAL second innings -- the real chasing XI, the real
bowling plan, the real target -- and ask "how often should this team have won?"
Simulate it `--sims` times, and the fraction of wins is the model's implied win
probability for that exact match. Compare against what actually happened.

This is the test that matters most to a player: an unwinnable chase that feels
winnable, or vice versa, is the first thing anyone notices. Every RRR / pressure
feature in the model exists specifically to get this right, and nothing else in
the harness checks it -- the full-innings metrics are dominated by first-innings
balls, where there is no target to chase at all.

Two numbers, standard for a probabilistic forecaster:
  Brier score  mean((p_win - actual)^2), lower is better. 0.25 is "always guess 50%".
  log-loss     -mean(log p(actual)), lower is better, penalises confident wrong calls hard.
Plus a reliability table: bucket matches by predicted win probability, check the
ACTUAL win rate in each bucket matches. A model that says "70%" should be right
about 70% of the time, not systematically 55% or 85%.
"""

from __future__ import annotations

import argparse
import math
import random

from ml.etl.replay import iter_innings
from ml.harness.run_model import model_ball_fn, model_extras_fn
from ml.harness.simulate import RoleMix, simulate_innings, venue_rates
from ml.runtime import players as P


def real_chases(season_min: int | None = None, limit: int | None = None):
    """Second innings with a real target, from real matches."""
    out = []
    for innings in iter_innings(limit=limit):
        if innings.innings_no != 2 or innings.target is None:
            continue
        if season_min is not None and innings.season < season_min:
            continue
        out.append(innings)
    return out


def simulate_chase(chase, model, calib, day_sigma, n_sims, seed, extras_fn, venue):
    by_name = P.load_players()
    lineup = [by_name[nm] for nm in chase.lineup if nm in by_name][:11]
    overs = [by_name[nm] for nm in chase.bowler_by_over if nm in by_name]
    if len(lineup) < 11 or len(overs) < 4:
        return None

    ball_fn = model_ball_fn(model, calibration=calib["calibration"],
                            out_calibration=calib["out_calibration"])
    league_avg = P.league_avg()
    v_rpb, v_wpb = venue.get(chase.venue, venue[None])
    rng = random.Random(seed)

    wins = 0
    for i in range(n_sims):
        random.seed(seed * 10_000 + i)     # calculate_single_ball's global RNG
        out = simulate_innings(
            lineup, overs, ball_fn, league_avg,
            target=chase.target, rng=rng, role_mix=RoleMix.neutral(),
            extras_fn=extras_fn, venue_rpb=v_rpb, venue_wpb=v_wpb,
            day_sigma=day_sigma,
        )
        if out.chased:
            wins += 1
    return wins / n_sims


def brier(pairs):
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def logloss(pairs, eps=1e-6):
    total = 0.0
    for p, y in pairs:
        p = min(max(p, eps), 1 - eps)
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(pairs)


def reliability_table(pairs, n_bins=5):
    pairs = sorted(pairs, key=lambda t: t[0])
    rows = []
    n = len(pairs)
    for i in range(n_bins):
        lo, hi = i * n // n_bins, (i + 1) * n // n_bins
        chunk = pairs[lo:hi]
        if not chunk:
            continue
        pred = sum(p for p, _ in chunk) / len(chunk)
        actual = sum(y for _, y in chunk) / len(chunk)
        rows.append((pred, actual, len(chunk)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=400,
                    help="real chases to evaluate (sampled evenly across all of them)")
    ap.add_argument("--sims", type=int, default=40,
                    help="simulations per chase -- the sample size behind each win%%")
    ap.add_argument("--since", type=int, default=2023,
                    help="only chases from this season on (0 = all-time)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from ml.harness.run_baseline import ARTIFACTS
    from ml.runtime.model import OutcomeModel
    import json
    import os

    calib_path = os.path.join(ARTIFACTS, "model_calibration.json")
    calib = {"day_sigma": 0.225, "calibration": 1.0, "out_calibration": 1.0}
    try:
        calib.update(json.load(open(calib_path, encoding="utf-8")))
    except OSError:
        pass

    model = OutcomeModel.load()
    all_chases = real_chases(season_min=args.since or None)
    if not all_chases:
        raise SystemExit("no chases found for the given --since filter")

    step = max(1, len(all_chases) // args.matches)
    chases = all_chases[::step][:args.matches]
    print(f"{len(all_chases)} real chases available"
          f"{f' (seasons >= {args.since})' if args.since else ''}, "
          f"evaluating {len(chases)} of them at {args.sims} sims each\n")

    venue = venue_rates(list(iter_innings(limit=None)))
    extras_fn = model_extras_fn(model)

    pairs = []       # (predicted p_win, actual 0/1)
    skipped = 0
    for i, chase in enumerate(chases):
        actual = 1 if chase.total >= chase.target else 0
        p = simulate_chase(chase, model, calib, calib["day_sigma"], args.sims,
                           seed=args.seed + i, extras_fn=extras_fn, venue=venue)
        if p is None:
            skipped += 1
            continue
        pairs.append((p, actual))
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(chases)}", flush=True)

    if not pairs:
        raise SystemExit("every chase was skipped (name-join failures?)")

    real_rate = sum(y for _, y in pairs) / len(pairs)
    mean_pred = sum(p for p, _ in pairs) / len(pairs)
    print(f"\n{len(pairs)} chases scored ({skipped} skipped -- incomplete XI/plan)")
    print(f"  actual chase success rate : {100 * real_rate:5.1f}%")
    print(f"  mean predicted win prob   : {100 * mean_pred:5.1f}%")
    print(f"  Brier score               : {brier(pairs):.4f}   "
          f"(0.25 = always guessing 50/50, lower is better)")
    print(f"  log-loss                  : {logloss(pairs):.4f}")

    print(f"\nreliability -- predicted vs ACTUAL win rate, grouped by predicted probability:")
    print(f"  {'predicted':>10} {'actual':>8} {'n':>5}")
    for pred, actual, n in reliability_table(pairs):
        flag = "" if abs(pred - actual) < 0.15 else "  <-- off"
        print(f"  {100 * pred:>9.1f}% {100 * actual:>7.1f}% {n:>5}{flag}")


if __name__ == "__main__":
    main()
