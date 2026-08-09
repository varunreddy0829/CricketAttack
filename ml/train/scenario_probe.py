"""Is the OVR measurement biased against anchors by only testing ONE scenario?

    ml/.venv/Scripts/python -m ml.train.scenario_probe --era 2014_2022

derive_ovr measures every player in a first innings, no target, slot #3, behind a
full-strength lineup. That is the situation in which an anchor is worth the
LEAST: nobody is under pressure, wickets are cheap, and pure scoring rate wins.

If a high-average, medium-tempo batter is genuinely valuable in the situations
anchors exist for -- rebuilding after a collapse, or holding a chase together --
then their measured value should climb in those scenarios relative to a pure
hitter. If it doesn't move, the low rating is real and not an artifact of how we
measure.

Scenarios, all paired on the same seeds so they're directly comparable:
  flat     first innings, no target, full lineup      (what derive_ovr uses)
  chase    a real target, so wickets and rate BOTH bite
  collapse the three batters above the slot replaced by tail-enders
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
from ml.train.derive_ovr import ERA_ROOT, _median_pool, _usable_plans

# derive_ovr now rotates the slot; this probe deliberately holds it fixed at #3
# so the scenario is the only thing changing.
BAT_SLOT = 2

PROBE = ["V Kohli", "DA Warner", "S Dhawan", "AB de Villiers", "AD Russell",
         "JC Buttler", "MS Dhoni", "F du Plessis", "KL Rahul", "SV Samson"]

CHASE_TARGET = 190          # a demanding but ordinary modern chase


def _weakest(records, n):
    """The n worst rateable batters -- stand-ins for a collapsed top order."""
    cands = [r for r in records if r.get("rateable_batting")]
    return sorted(cands, key=lambda r: r["batting"]["sr"])[:n]


def run(era: E.Era, n: int, seed: int) -> None:
    path = os.path.join(ERA_ROOT, era.id, "players.json")
    with open(path, "r", encoding="utf-8") as fh:
        records = json.load(fh)
    by_name = {r["name"]: r for r in records}
    for r in records:
        r["_bat_rank"] = r["batting"]["sr"] * (r["batting"]["balls"] ** 0.25)

    _, plans = real_reference(era=era)
    usable = _usable_plans(plans, by_name)
    model = OutcomeModel.load(era_id=era.id)
    cal = load_calibration(era_id=era.id)
    league = P.league_avg()
    ball_fn = model_ball_fn(model, calibration=cal["calibration"],
                            out_calibration=cal["out_calibration"])
    extras = model_extras_fn(model)
    mix = RoleMix.realistic()
    tail = _weakest(records, BAT_SLOT)

    def team_total(record, scenario) -> float:
        total = 0.0
        for i in range(n):
            lu_src, ov_src = usable[i % len(usable)]
            lu = list(lu_src)
            lu[BAT_SLOT] = record
            target = None
            if scenario == "collapse":
                # gut the top order so the slot walks in at a genuine crisis
                for s in range(BAT_SLOT):
                    lu[s] = tail[s]
            elif scenario == "chase":
                target = CHASE_TARGET
            rng = random.Random(seed * 1_000_003 + i)
            random.seed(seed * 7_919 + i)
            o = simulate_innings(lu, ov_src, ball_fn, league, target=target, rng=rng,
                                 role_mix=mix, extras_fn=extras,
                                 venue_rpb=1.4, venue_wpb=0.05,
                                 day_sigma=cal["day_sigma"])
            total += o.total
        return total / n

    baseline = _median_pool([r for r in records if r.get("rateable_batting")],
                            "_bat_rank")

    print(f"{era.id}: value added over a median batter, by scenario "
          f"({n} paired innings each)")
    print(f"  chase target {CHASE_TARGET}; collapse = top {BAT_SLOT} replaced by "
          f"the era's weakest bats\n")
    print(f"  {'player':<20} {'SR':>7} {'avg':>6} | {'flat':>7} {'chase':>7} "
          f"{'collapse':>9} | {'shift':>7}")
    print("  " + "-" * 74)

    base = {s: team_total(baseline, s) for s in ("flat", "chase", "collapse")}
    rows = []
    for name in PROBE:
        rec = by_name.get(name)
        if not rec or not rec.get("rateable_batting"):
            continue
        v = {s: team_total(rec, s) - base[s] for s in ("flat", "chase", "collapse")}
        b = rec["batting"]
        # does this player gain ground when the situation gets hard?
        shift = ((v["chase"] + v["collapse"]) / 2) - v["flat"]
        rows.append((name, b["sr"], b["avg"], v, shift, rec["batting_ovr"]))

    for name, sr, avg, v, shift, ovr in rows:
        print(f"  {name:<20} {sr:>7} {avg:>6} | {v['flat']:>+7.1f} "
              f"{v['chase']:>+7.1f} {v['collapse']:>+9.1f} | {shift:>+7.1f}")

    print()
    anchors = [r for r in rows if r[1] < 140]
    hitters = [r for r in rows if r[1] >= 140]
    for label, grp in (("anchors (SR<140)", anchors), ("hitters (SR>=140)", hitters)):
        if grp:
            print(f"  mean shift, {label:<18} {sum(r[4] for r in grp) / len(grp):>+6.2f} runs")
    print("\n  A positive shift means the player gains value when the situation is")
    print("  hard. If anchors don't shift up, their low flat rating is real.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--era", default="2014_2022")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    run(E.get(args.era), args.n, args.seed)


if __name__ == "__main__":
    main()
