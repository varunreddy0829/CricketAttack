"""Derive OVRs from the GRID raters, and write them into the era's player pool.

    ml/.venv/Scripts/python -m ml.train.derive_ovr_grid --era 2014_2022

Replaces ml/train/derive_ovr.py's measurement. Same 55-99 band and the same
median->68 / p95->88 anchoring, so draft_generator's tier cuts keep their meaning;
only the number being rescaled changes.

## What each side is rated on

  BATTING   expected runs from a ball allowance sampled, per position, from real
            innings -- ml/grid_rate.py
  BOWLING   runs conceded minus wickets x their measured cost, over a real 4-over
            spell against a declining batting order -- ml/grid_rate_bowl.py

Both walk an enumerated grid of grounds x phase x opposition x innings with every
remaining input drawn from real balls, so no situation is pinned to a value
somebody chose. Both read the model's probabilities directly rather than sampling
outcomes, so there is no Monte Carlo noise in the ratings.

Both also run with the engine's live settings -- shrinkage toward replacement and
the longevity contest -- so an OVR is what the player is worth IN THE GAME, not
what he was worth in an engine nobody plays.

## Why bowling inverts

The bowling metric is a cost: fewer runs is better. It is negated before the
rescale so that, on both sides, bigger is better and the band means the same
thing.
"""

from __future__ import annotations

import argparse
import json
import os

from ml.etl import eras as E
from ml.runtime import longevity as L
from ml.runtime.engine import load_calibration
from ml.runtime.model import SHRINK_BALLS, SHRINK_TARGET
from ml.train.derive_ovr import ERA_ROOT, to_ovr


def _shrink_k(era):
    """That era's own fitted K -- 2023-2026 takes 250 where the others take
    500, because it drives the all-out rate and each era's tail differs."""
    return load_calibration(era.id).get('shrink_balls', SHRINK_BALLS)


def _rate_batting(era, reps):
    from ml.grid_rate import rate
    rows = rate(era, reps=reps, dial=L.LONGEVITY_DIAL,
                volume_weight=L.VOLUME_WEIGHT, longevity_on=True, seed=11,
                shrink=_shrink_k(era), target=SHRINK_TARGET)
    return {r["name"]: r["expected"] for r in rows}


def _rate_bowling(era, reps):
    from ml.grid_rate_bowl import rate, wicket_value
    rows = rate(era, reps=reps, dial=L.LONGEVITY_DIAL,
                volume_weight=L.VOLUME_WEIGHT, longevity_on=True, seed=11,
                shrink=_shrink_k(era), target=SHRINK_TARGET)
    wv = wicket_value(era)
    # negated: the metric is a COST, and the band must mean "bigger is better"
    return {r["name"]: -(r["runs"] - wv * r["wkts"]) for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--era", default="2014_2022")
    ap.add_argument("--bat-reps", type=int, default=60)
    ap.add_argument("--bowl-reps", type=int, default=25)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    era = E.get(args.era)
    print(f"\n{era.id}  {era.label}")
    print(f"  engine settings: shrink {_shrink_k(era):g} -> {SHRINK_TARGET}, "
          f"longevity dial {L.LONGEVITY_DIAL}, W {L.VOLUME_WEIGHT}\n")

    print("[1/2] batting grid ...", flush=True)
    bat = _rate_batting(era, args.bat_reps)
    print("\n[2/2] bowling grid ...", flush=True)
    bowl = _rate_bowling(era, args.bowl_reps)

    bat_ovr, bowl_ovr = to_ovr(bat), to_ovr(bowl)

    path = os.path.join(ERA_ROOT, era.id, "players.json")
    with open(path, "r", encoding="utf-8") as fh:
        records = json.load(fh)
    for r in records:
        r["batting_ovr"] = bat_ovr.get(r["name"])
        r["bowling_ovr"] = bowl_ovr.get(r["name"])
        r["measured_bat_value"] = round(bat[r["name"]], 3) if r["name"] in bat else None
        r["measured_bowl_value"] = round(bowl[r["name"]], 3) if r["name"] in bowl else None

    if not args.dry_run:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=1)
        print(f"\n  wrote {path}")
    else:
        print("\n  dry run -- nothing written")

    for label, key in (("BATTING", "batting_ovr"), ("BOWLING", "bowling_ovr")):
        top = sorted((r for r in records if r.get(key)),
                     key=lambda r: -r[key])[:12]
        rated = [r[key] for r in records if r.get(key)]
        print(f"\n  TOP 12 {label}   ({len(rated)} rated, "
              f"median {sorted(rated)[len(rated) // 2]}, max {max(rated)})")
        for r in top:
            b = r["batting"] if key == "batting_ovr" else r["bowling"]
            vol = b.get("balls", b.get("legal_balls", 0))
            print(f"    {r[key]:>3}  {r['name']:<24}{vol:>6} balls")


if __name__ == "__main__":
    main()
