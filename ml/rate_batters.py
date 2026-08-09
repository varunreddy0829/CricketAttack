"""What does the trained model think each batter actually DOES?

    ml/.venv/Scripts/python -m ml.rate_batters --era 2014_2022 --top 30

Not OVR, and not team-runs-added. This is the plainest question you can ask the
model: put this player in, let him bat, and see how many he scores and how fast.

Method, and the parts that matter:

  SAME SLOT, SAME TEAM-MATES.  Every player bats at the same position behind the
  same lineup against the same bowling plans, so the comparison is between the
  PLAYERS and not between the situations they happened to walk into. Real career
  figures cannot do this -- an opener faces the new ball and a number 7 faces the
  death, and neither number is a like-for-like read on the player.

  COMMON RANDOM NUMBERS.  Every player is measured on the same innings seeds. The
  spread between adjacent players is small, and unpaired noise would be larger
  than the signal.

  HIS OWN INNINGS, not the team's.  Runs and balls are read off the striker's own
  scorecard, so 40 off 20 and 40 off 35 are correctly different players.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random

from ml.etl import eras as E
from ml.harness.run_baseline import real_reference
from ml.harness.run_model import model_ball_fn, model_extras_fn
from ml.harness.simulate import RoleMix, simulate_innings
from ml.runtime import players as P
from ml.runtime.engine import load_calibration
from ml.runtime.model import OutcomeModel
from ml.runtime import longevity as L
from ml.runtime.longevity import apply_longevity, build_scores
from ml.train.derive_ovr import _usable_plans, ERA_ROOT

# Rotate the batting slot AND the ground rather than pinning both. Fixing them
# is what makes a comparison fair, but fixing them at ONE value answers a much
# narrower question -- "who is best at #4 on a neutral pitch" is not the same as
# "who is the better player". Every batter sees the SAME rotation in the SAME
# order with the SAME seeds, so the pairing that removes noise is untouched.
SLOTS = (0, 1, 2, 3, 4, 5, 6)          # opener through number 7

# Runs-per-ball standard deviation for a T20 batter, across outcomes 0/1/2/4/6.
# Used to size how UNCERTAIN a player's rating is, given how much of him we saw.
RUNS_PER_BALL_SD = 1.55

# How many standard errors to subtract when ranking (see confidence_penalty).
# 1.0 is a one-sigma lower bound: rank a player by what we are reasonably sure
# he is worth, not by the midpoint of a wide guess.
CONFIDENCE_LAMBDA = 1.0


def confidence_penalty(balls: int, innings_balls: float,
                       lam: float = CONFIDENCE_LAMBDA) -> float:
    """Runs to deduct for not knowing how good this player is.

    T Curran's rating rests on 104 career balls and S Dhawan's on 3245, and a
    plain ranking treats those as equally trustworthy -- which is how a man with
    126 career runs finished above one with 4182. The error on a rate estimate
    falls as 1/sqrt(n), so the fix is to rank by a LOWER CONFIDENCE BOUND rather
    than the point estimate.

    This is not a reputation bonus and not a longevity bonus. It makes no claim
    that a long career deserves reward; it says only that a short one is not yet
    evidence. A 104-ball player and a 3245-ball player differ by ~2.5 runs over a
    20-ball innings, which is exactly the size of the gap that had them inverted.
    """
    if balls <= 0:
        return lam * RUNS_PER_BALL_SD * innings_balls
    return lam * (RUNS_PER_BALL_SD / math.sqrt(balls)) * innings_balls


def seasons_batted(era: E.Era) -> dict[str, int]:
    """name -> how many separate seasons he actually batted in, this era."""
    from collections import defaultdict

    from ml.etl.replay import iter_innings
    seen = defaultdict(set)
    for inn in iter_innings():
        if not era.covers(inn.season):
            continue
        for b in inn.balls:
            if b.outcome != "wide":
                seen[b.batter].add(inn.season)
    return {k: len(v) for k, v in seen.items()}


def _venues(era: E.Era):
    """Every ground with a measured profile, so scoring level, boundary share and
    the spin/pace edge all vary the way they do across a real season."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "ml", "artifacts", "eras", era.id, "venue_profile.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            g = json.load(fh)["grounds"]
    except OSError:
        return [(1.345, 0.0485, 0.589, 0.0, 0.0)]
    out = []
    for v in g.values():
        out.append((v["rpb"], v["wpb"], v["bdry_share"],
                    v.get("spin_edge") or 0.0, v.get("pace_edge") or 0.0))
    return sorted(out)


def rate(era: E.Era, n: int, seed: int, slot: int, *,
         longevity_on: bool = True, volume_weight: float = 0.5,
         dial: float = L.LONGEVITY_DIAL, curve: str = "squared") -> list[dict]:
    path = os.path.join(ERA_ROOT, era.id, "players.json")
    with open(path, "r", encoding="utf-8") as fh:
        records = json.load(fh)
    by_name = {r["name"]: r for r in records}

    _, plans = real_reference(era=era)
    usable = _usable_plans(plans, by_name)
    if not usable:
        raise SystemExit(f"{era.id}: no usable lineups")

    model = OutcomeModel.load(era_id=era.id)
    cal = load_calibration(era_id=era.id)
    league = P.league_avg()
    ball_fn = model_ball_fn(model, calibration=cal["calibration"],
                            out_calibration=cal["out_calibration"])
    extras = model_extras_fn(model)
    mix = RoleMix.realistic()

    # the longevity layer, applied to the model's gold probs exactly as roles are
    scores = build_scores(records, volume_weight=volume_weight)
    base_ball_fn = ball_fn
    if longevity_on:
        import random as _r
        from ml.runtime.model import ENGINE_KEYS as _K

        def ball_fn(striker, bowler, league_avg, context=None):   # noqa: F811
            w = model.predict(striker, bowler, context or {})[0]
            bs = scores["bat"].get(striker.name, 0.0)
            ws = scores["bowl"].get(bowler.name, 0.0)
            if curve == "linear":
                # apply_longevity SQUARES whatever difference it is handed, so
                # pre-take the root: sign(d)*sqrt(|d|) squares back to exactly d,
                # making the transfer linear in the score gap.
                d = max(-1.0, min(1.0, bs - ws))
                root = (1.0 if d >= 0 else -1.0) * abs(d) ** 0.5
                w = apply_longevity(w, root, 0.0, dial=dial)
            else:
                w = apply_longevity(w, bs, ws, dial=dial)
            w = {k: v * (cal["calibration"] if k not in ("0", "Out") else 1.0)
                 for k, v in w.items()}
            tot = sum(w.values()) or 1.0
            return _r.choices(list(w), weights=[v / tot for v in w.values()], k=1)[0]

    seasons = seasons_batted(era)
    venues = _venues(era)
    cands = [r for r in records if r.get("rateable_batting")]
    print(f"  {len(cands)} batters x {n} innings each")
    print(f"  rotating slots {tuple(x + 1 for x in SLOTS)} x {len(venues)} grounds "
          f"x {min(n, 1079)} real attacks -- identical rotation for every player")

    out = []
    for i, rec in enumerate(cands, 1):
        runs = balls = outs = inns = 0
        for k in range(n):
            lu_src, ov_src = usable[k % len(usable)]
            lu = list(lu_src)
            slot = SLOTS[k % len(SLOTS)]
            v_rpb, v_wpb, v_bdry, v_spin, v_pace = venues[k % len(venues)]
            lu[slot] = rec
            rng = random.Random(seed * 1_000_003 + k)
            random.seed(seed * 7_919 + k)
            o = simulate_innings(lu, ov_src, ball_fn, league, target=None, rng=rng,
                                 role_mix=mix, extras_fn=extras,
                                 venue_rpb=v_rpb, venue_wpb=v_wpb,
                                 venue_bdry_share=v_bdry,
                                 venue_spin_edge=v_spin, venue_pace_edge=v_pace,
                                 day_sigma=cal["day_sigma"])
            if slot < len(o.batter_scores):
                runs += o.batter_scores[slot]
                balls += o.batter_balls[slot]
                inns += 1
                # he was dismissed if someone below him came in, or the innings
                # ended with him not at the crease
                outs += 1 if len(o.batter_scores) > slot + 2 else 0
        if not inns or not balls:
            continue
        out.append({
            "name": rec["name"],
            "runs": runs / inns,
            "sr": 100.0 * runs / balls,
            "balls": balls / inns,
            "real_sr": rec["batting"]["sr"],
            "real_runs": rec["batting"]["runs"],
            "real_balls": rec["batting"]["balls"],
            "seasons": seasons.get(rec["name"], 0),
            "penalty": confidence_penalty(rec["batting"]["balls"], balls / inns),
            "lscore": scores["bat"].get(rec["name"], 0.0),
        })
        if i % 40 == 0:
            print(f"    {i}/{len(cands)}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--era", default="2014_2022")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--slot", type=int, default=3)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--sort", default="runs", choices=("runs", "sr"))
    ap.add_argument("--w", type=float, default=0.5,
                    help="longevity slider: 1=career volume only, 0=quality only")
    ap.add_argument("--no-longevity", action="store_true")
    ap.add_argument("--dial", type=float, default=L.LONGEVITY_DIAL,
                    help="strength of the longevity transfer")
    ap.add_argument("--curve", default="squared", choices=("squared", "linear"))
    ap.add_argument("--confidence", type=float, default=0.0,
                    help="lower-confidence-bound strength; 0 = raw point estimate")
    args = ap.parse_args()

    era = E.get(args.era)
    print(f"\n{era.id}  {era.label}\n")
    rows = rate(era, args.n, args.seed, args.slot,
                longevity_on=not args.no_longevity, volume_weight=args.w,
                dial=args.dial, curve=args.curve)
    for r in rows:
        r["adj"] = r["runs"] - r["penalty"] * args.confidence
    key = "adj" if args.sort == "runs" else args.sort
    rows.sort(key=lambda r: -r[key])

    print(f"\n  TOP {args.top} BATTERS -- rotated over slots 1-7 x 15 grounds")
    print(f"  {'#':>3}  {'batter':<22}| {'MODEL (rotated)':>16} "
          f"|{'  ACTUAL 2014-2022 CAREER':<34}")
    print(f"  {'':>3}  {'':<22}| {'runs':>6}{'balls':>7}{'SR':>7} "
          f"|{'runs':>8}{'seasons':>9}{'SR':>8} | {'L-score':>8}")
    print("  " + "-" * 92)
    for i, r in enumerate(rows[:args.top], 1):
        print(f"  {i:>3}  {r['name']:<22}| {r['runs']:>6.1f}{r['balls']:>7.1f}"
              f"{r['sr']:>7.1f} |{r['real_runs']:>8}{r['seasons']:>9}"
              f"{r['real_sr']:>8.1f} | {r['lscore']:>+8.2f}")


if __name__ == "__main__":
    main()
