"""Phase C: MEASURE each player's OVR by simulating them through their era's model.

    ml/.venv/Scripts/python -m ml.train.derive_ovr --era all

Writes `batting_ovr` / `bowling_ovr` back into data/eras/<id>/players.json.

## Why this exists

The old OVR came from a hand-written power formula over career stats. Measured
against what players actually deliver, it is barely correlated with value: on the
career-wide model the four highest-rated players (99, 98, 95, 93) ranked 4th,
5th, 9th and 7th by team runs added, and a 93-rated Gayle added fewer runs than
an 80-rated Abhishek Sharma. An auction priced on that number charges 17cr for
something worth 8.

So OVR is no longer computed. It is measured: put the player in a real lineup,
simulate, and see what the team scores.

## The metric, and the one that looked right but wasn't

**Team runs added over a replacement-level player.** For bowlers, runs prevented.

An earlier attempt scored batters on expected runs before dismissal -- which is
just batting average. It ranked Amla (SR 119) 4th and Abhishek Sharma (SR 143)
69th, because in T20 the scarce resource is BALLS, not wickets. A player who
scores 35 off 20 is worth more than one who scores 45 off 40, and any
average-shaped metric says the opposite. Team runs added gets this right because
the simulation itself is subject to the 120-ball limit.

## Paired comparison

Every player is measured on the SAME innings: same seed, same lineups, same
bowling plans, same day factors. Without common random numbers the ranking is
dominated by simulation noise -- the spread between two adjacent players is a
couple of runs, and unpaired noise at n=800 is larger than that.
"""

from __future__ import annotations

import argparse
import json
import os
import random

from ml.etl import eras as E
from ml.harness.run_baseline import real_reference
from ml.harness.run_model import model_ball_fn, model_extras_fn
from ml.harness.simulate import RoleMix, simulate_innings
from ml.runtime import players as P
from ml.runtime.engine import load_calibration
from ml.runtime.model import OutcomeModel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ERA_ROOT = os.path.join(REPO_ROOT, "data", "eras")

# OVR band the rest of the game already speaks: draft_generator's MARQUEE_CUT
# (80-85) and MID_FLOOR (70) are expressed on it, as are the UI's tier borders.
OVR_MIN, OVR_MAX = 55, 99

# The two anchor points the 55-99 band is pinned to (see to_ovr). MEDIAN_OVR
# sits just under draft_generator's MID_FLOOR(70) so the typical player is a
# base buy, and TOP_OVR is set so MARQUEE_CUT(80) lands around the top 15%.
MEDIAN_OVR, TOP_QUANTILE, TOP_OVR = 68, 0.95, 88

BAT_SLOT = 2        # measure batters at #3 -- reached in almost every innings
BOWL_SLOT = 0       # measure bowlers as the first-change option


def _usable_plans(plans, by_name):
    out = []
    for p in plans:
        lu = [by_name[n] for n in p.lineup if n in by_name][:11]
        ov = [by_name[n] for n in p.bowler_by_over if n in by_name]
        if len(lu) == 11 and len(ov) >= 4:
            out.append((lu, ov))
    return out


def _median_pool(pool: list, key: str) -> dict:
    """The replacement-level player: the pool's median rateable performer. Using
    a median rather than the worst keeps the baseline stable -- a single freak
    low-sample player would otherwise stretch every rating above it."""
    ranked = sorted(pool, key=lambda r: r[key])
    return ranked[len(ranked) // 2]


def measure_batting(era, model, cal, plans_by, n: int, seed: int) -> dict:
    """name -> team runs added over the replacement batter, at slot #3."""
    by_name, usable = plans_by
    league = P.league_avg()
    ball_fn = model_ball_fn(model, calibration=cal["calibration"],
                            out_calibration=cal["out_calibration"])
    extras = model_extras_fn(model)
    mix = RoleMix.realistic()

    cands = [r for r in by_name.values() if r.get("rateable_batting")]
    baseline = _median_pool(cands, "_bat_rank")

    def team_total(record) -> float:
        lu_base, ov = usable[0]
        total = 0.0
        for i in range(n):
            lu_src, ov_src = usable[i % len(usable)]
            lu = list(lu_src)
            lu[BAT_SLOT] = record          # same slot, same team-mates, every time
            # common random numbers: identical stream per innings index for
            # every player, so the comparison is paired
            rng = random.Random(seed * 1_000_003 + i)
            random.seed(seed * 7_919 + i)
            o = simulate_innings(lu, ov_src, ball_fn, league, target=None, rng=rng,
                                 role_mix=mix, extras_fn=extras,
                                 venue_rpb=1.4, venue_wpb=0.05,
                                 day_sigma=cal["day_sigma"])
            total += o.total
        return total / n

    base = team_total(baseline)
    out = {}
    for i, r in enumerate(cands, 1):
        out[r["name"]] = team_total(r) - base
        if i % 25 == 0:
            print(f"      batting {i}/{len(cands)}", flush=True)
    return out


def measure_bowling(era, model, cal, plans_by, n: int, seed: int) -> dict:
    """name -> runs PREVENTED vs the replacement bowler (higher = better)."""
    by_name, usable = plans_by
    league = P.league_avg()
    ball_fn = model_ball_fn(model, calibration=cal["calibration"],
                            out_calibration=cal["out_calibration"])
    extras = model_extras_fn(model)
    mix = RoleMix.realistic()

    cands = [r for r in by_name.values() if r.get("rateable_bowling")]
    baseline = _median_pool(cands, "_bowl_rank")

    def conceded(record) -> float:
        total = 0.0
        for i in range(n):
            lu, ov_src = usable[i % len(usable)]
            ov = list(ov_src)
            # give the candidate a full quota-sized share of the attack
            for slot in range(0, len(ov), max(1, len(ov) // 4)):
                ov[slot] = record
            rng = random.Random(seed * 1_000_003 + i)
            random.seed(seed * 7_919 + i)
            o = simulate_innings(lu, ov, ball_fn, league, target=None, rng=rng,
                                 role_mix=mix, extras_fn=extras,
                                 venue_rpb=1.4, venue_wpb=0.05,
                                 day_sigma=cal["day_sigma"])
            total += o.total
        return total / n

    base = conceded(baseline)
    out = {}
    for i, r in enumerate(cands, 1):
        out[r["name"]] = base - conceded(r)      # prevented runs
        if i % 25 == 0:
            print(f"      bowling {i}/{len(cands)}", flush=True)
    return out


def to_ovr(values: dict) -> dict:
    """Rescale measured value onto the 55-99 band, anchored on the pool's MEDIAN
    and its 95th percentile rather than its worst and best.

    Still linear in runs, deliberately: the measurement is already in the units
    the game cares about, so a rank transform would throw away the fact that the
    gap between the top two players is smaller than the gap between the bottom
    two. Only the two anchor points changed.

    Anchoring on min/max looked equivalent and was not. It hands the whole scale
    to the two most extreme players, so the tiers land wherever the tails happen
    to fall -- and the bowling distribution has a long thin BAD tail (a handful
    of part-timers who concede 12+ runs more than the baseline) against a tightly
    packed good end. In 2023-2026 that stretched the scale so far that the MEDIAN
    bowler rescaled to OVR 86, putting 69 of 100 bowlers above MARQUEE_CUT. An
    auction where two thirds of the pool is marquee has no scarcity and no tiers.

    Median -> 68 and p95 -> 88 fixes the tiers to what they're supposed to mean:
    the typical player sits just under MID_FLOOR, and marquee is the top ~15%.
    Both ends clip, so the bad tail compresses into 55 instead of setting the
    scale for everyone above it."""
    if not values:
        return {}
    ranked = sorted(values.values())
    n = len(ranked)
    mid = ranked[n // 2]
    top = ranked[min(n - 1, int(n * TOP_QUANTILE))]
    span = top - mid
    if span <= 0:
        return {k: MEDIAN_OVR for k in values}
    return {k: max(OVR_MIN, min(OVR_MAX,
                                int(round(MEDIAN_OVR + (v - mid) / span
                                          * (TOP_OVR - MEDIAN_OVR)))))
            for k, v in values.items()}


def rescale_only(era: E.Era) -> None:
    """Re-run just to_ovr over the ALREADY-measured values on disk.

    The simulation is the expensive half (~40 min an era) and its output --
    `measured_bat_value` / `measured_bowl_value`, in runs -- is what gets stored.
    Changing how those runs map onto the 55-99 band costs nothing and needs no
    re-measurement, so tuning the anchors doesn't mean re-simulating."""
    path = os.path.join(ERA_ROOT, era.id, "players.json")
    with open(path, "r", encoding="utf-8") as fh:
        records = json.load(fh)

    for key, mkey, flag in (("batting_ovr", "measured_bat_value", "rateable_batting"),
                            ("bowling_ovr", "measured_bowl_value", "rateable_bowling")):
        vals = {r["name"]: r[mkey] for r in records
                if r.get(flag) and mkey in r}
        if not vals:
            print(f"  {era.id}: no {mkey} on disk -- run the full derivation first")
            continue
        ovr = to_ovr(vals)
        for r in records:
            r[key] = ovr.get(r["name"])
        band = sorted(ovr.values())
        marquee = sum(1 for v in band if v >= 80)
        print(f"  {era.id:<10} {key:<12} med {band[len(band) // 2]:>3}  "
              f"marquee {marquee:>3}/{len(band)}")

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=1)


def derive(era: E.Era, n: int, seed: int) -> None:
    print(f"\n{'=' * 62}\n{era.id}  ({era.first}-{era.last})  {era.label}\n{'=' * 62}")
    path = os.path.join(ERA_ROOT, era.id, "players.json")
    with open(path, "r", encoding="utf-8") as fh:
        records = json.load(fh)
    by_name = {r["name"]: r for r in records}

    # provisional ranks only so the replacement baseline is reproducible
    for r in records:
        r["_bat_rank"] = r["batting"]["sr"] * (r["batting"]["balls"] ** 0.25)
        r["_bowl_rank"] = -r["bowling"]["eco"] * (r["bowling"]["legal_balls"] ** 0.25)

    _, plans = real_reference(era=era)
    usable = _usable_plans(plans, by_name)
    if not usable:
        raise SystemExit(f"{era.id}: no usable lineups -- name join failed")
    print(f"  {len(usable)} real lineups, {n} innings per player")

    model = OutcomeModel.load(era_id=era.id)
    cal = load_calibration(era_id=era.id)
    ctx = (by_name, usable)

    bat = measure_batting(era, model, cal, ctx, n, seed)
    bowl = measure_bowling(era, model, cal, ctx, n, seed)
    bat_ovr, bowl_ovr = to_ovr(bat), to_ovr(bowl)

    for r in records:
        r.pop("_bat_rank", None)
        r.pop("_bowl_rank", None)
        r["batting_ovr"] = bat_ovr.get(r["name"])
        r["bowling_ovr"] = bowl_ovr.get(r["name"])
        r["measured_bat_value"] = round(bat.get(r["name"], 0.0), 2)
        r["measured_bowl_value"] = round(bowl.get(r["name"], 0.0), 2)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=1)

    top = sorted((r for r in records if r["batting_ovr"]),
                 key=lambda r: -r["batting_ovr"])[:8]
    print(f"\n  top batters by MEASURED value:")
    for r in top:
        print(f"    {r['batting_ovr']:>3}  {r['name']:<24}"
              f"{r['measured_bat_value']:>+7.1f} runs   SR {r['batting']['sr']:>6}")
    print(f"\n  wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--era", default="all")
    ap.add_argument("--n", type=int, default=400,
                    help="innings per player. Paired sampling makes this go a long "
                         "way; 400 resolves differences of ~1.5 runs.")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--rescale-only", action="store_true",
                    help="skip the simulation; just remap the measured runs "
                         "already on disk onto the OVR band")
    args = ap.parse_args()
    targets = E.MODEL_ERAS if args.era == "all" else [E.get(args.era)]
    for era in targets:
        if args.rescale_only:
            rescale_only(era)
        else:
            derive(era, args.n, args.seed)


if __name__ == "__main__":
    main()
