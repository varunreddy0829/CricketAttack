"""Compute REAL per-ground scoring rates for the game's configured stadiums.

    ml/.venv/Scripts/python -m ml.etl.compute_venue_stats [--since 2023]

Writes ml/runtime/venue_stats.json -- a small, checked-in lookup table so live
play can hand the model the ACTUAL ground's real runs/ball and wickets/ball,
instead of the fixed league-average placeholder it uses today.

`--since` restricts the window to recent seasons, default 2023 (matching the
`--since 2023` convention already used in ml/harness/run_model.py and
ml/harness/calibrate_variance.py). This is a deliberate choice, not just less
data for its own sake: it puts every ground on comparable footing regardless of
how long it's existed (a venue that entered the IPL in 2023 would otherwise look
artificially "thin" next to one with 16 seasons of history, even though the real
difference is age, not reliability), and it matches how a player actually
remembers a ground -- as it plays now, not as an all-time average across pitches
that may have been re-laid or renovated since.

Unlike the training features (ml/etl/build_table.py), this is NOT leave-one-
match-out: there is no "current match" to exclude during live play, we just want
the best available real estimate for each ground. Every Cricsheet name variant
for the same physical ground (see ml/runtime/venues.py) is pooled together, so a
ground doesn't get a weaker estimate just because its name was recorded three
different ways across the years.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from ml.etl.replay import iter_innings
from ml.runtime.venues import CANONICAL_GROUNDS, canonical_ground

OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "runtime", "venue_stats.json")

# Below this, the ground's own (windowed) history is too thin to trust and it
# falls back to the league average instead. At a 2023+ window most grounds land
# at 4,800-7,600 balls; PCA Mohali is the outlier at ~940 (its home team, Punjab
# Kings, played its recent home games elsewhere). 2,000 clears every real ground
# and excludes only that one.
MIN_BALLS = 2000


def compute(season_min: int | None = None, season_max: int | None = None) -> dict:
    agg: dict = defaultdict(lambda: [0, 0, 0])     # balls, runs, wickets
    tot = [0, 0, 0]
    raw_names_seen: dict = defaultdict(set)

    for innings in iter_innings():
        if season_min is not None and innings.season < season_min:
            continue
        if season_max is not None and innings.season > season_max:
            continue
        key = canonical_ground(innings.venue)
        for b in innings.balls:
            if not b.is_legal:
                continue
            tot[0] += 1
            tot[1] += b.runs_total
            tot[2] += 1 if b.outcome == "Out" else 0
            if key:
                a = agg[key]
                a[0] += 1
                a[1] += b.runs_total
                a[2] += 1 if b.outcome == "Out" else 0
                raw_names_seen[key].add(innings.venue)

    league_rpb = tot[1] / max(1, tot[0])
    league_wpb = tot[2] / max(1, tot[0])

    out = {
        "_league_fallback": {
            "runs_per_ball": round(league_rpb, 4),
            "wkts_per_ball": round(league_wpb, 5),
        },
        "_season_min": season_min,
        "_season_max": season_max,
    }
    for key in CANONICAL_GROUNDS:
        balls, runs, wkts = agg.get(key, [0, 0, 0])
        entry = {"balls": balls, "name_variants": sorted(raw_names_seen.get(key, []))}
        if balls >= MIN_BALLS:
            entry["runs_per_ball"] = round(runs / balls, 4)
            entry["wkts_per_ball"] = round(wkts / balls, 5)
        else:
            entry["runs_per_ball"] = round(league_rpb, 4)
            entry["wkts_per_ball"] = round(league_wpb, 5)
            entry["note"] = f"only {balls} real balls -- league average used instead"
        out[key] = entry
    return out


def _report(data: dict, path: str, label: str) -> None:
    print(f"wrote {path}  ({label})")
    fb = data["_league_fallback"]
    print(f"  {'(league avg)':<16} {'':>7} {fb['runs_per_ball']:>10.4f} "
          f"{fb['wkts_per_ball']:>10.5f}")
    for key in CANONICAL_GROUNDS:
        e = data[key]
        flag = "  <- league fallback" if "note" in e else ""
        print(f"  {key:<16} {e['balls']:>7} {e['runs_per_ball']:>10.4f} "
              f"{e['wkts_per_ball']:>10.5f}{flag}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eras", action="store_true",
                    help="write per-era venue stats for every model era. Grounds "
                         "scored very differently across eras, so reusing one "
                         "window's rates in another double-counts the same way "
                         "the classic pitch multipliers did.")
    ap.add_argument("--since", type=int, default=2023,
                    help="single-window mode: only seasons >= this. 0 = all-time.")
    args = ap.parse_args()

    if args.eras:
        from ml.etl import eras as E
        artifacts = os.path.dirname(os.path.dirname(OUT_PATH))
        # The multiverse is included even though it is not a MODEL_ERA in the
        # training sense: it now has its own tagged model, whose ball table was
        # built with venue aggregates over ALL THREE eras, so it needs matching
        # all-era venue stats at serve time or train and serve disagree about
        # what a ground scores.
        targets = list(E.MODEL_ERAS)
        mv = E.get(E.MULTIVERSE)
        if mv not in targets:
            targets.append(mv)
        for era in targets:
            data = compute(season_min=era.first, season_max=era.last)
            d = os.path.join(artifacts, "artifacts", "eras", era.id)
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, "venue_stats.json")
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            _report(data, p, f"{era.first}-{era.last}")
    else:
        data = compute(season_min=args.since or None)
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        _report(data, OUT_PATH,
                f"seasons >= {args.since}" if args.since else "all-time")
