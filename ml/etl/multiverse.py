"""The multiverse pool: every era's version of a player, all at once.

    ml/.venv/Scripts/python -m ml.etl.multiverse

2008 Gayle, 2016 Gayle and 2024 Gayle are genuinely different cricketers -- the
whole point of splitting the game by era. This pool puts all of them on the same
auction floor, so you can buy the version you want and field a side drawn from
three decades.

## Names

Each entry is tagged with its era name: "CH Gayle (Genesis)". The tag is the
primary key, and `source_era` records where he came from, so the UI can show
both. The three names are Genesis, The Shift and Modern Era.

## Which model plays the ball

One era's, necessarily -- a delivery has a batter and a bowler who may come from
different decades, and there is no coherent way to run two models on one ball.
The middle era is the natural choice: it sits between the other two, so neither
end is played under rules wildly foreign to it.

That is why the players' own STATS still matter. The model reads a batter's
strike rate, average, boundary rates and playstyle grids as inputs, so 2008 Gayle
brings his 2008 numbers and behaves like 2008 Gayle even under the middle-era
model. What changes is the surrounding game, not the man.

## The cold-start that makes it work

Every tagged name is a stranger to that model, and an unrecognised name normally
contributes a ZERO player effect -- which would make all 900 entries play
identically. ml/runtime/model.py::cold_effect projects each player's own
observables through the shared W instead (E = A.W, without the learned
per-player D), which is precisely the cold start the low-rank design exists for.
"""

from __future__ import annotations

import json
import os

from ml.etl import eras as E

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ERA_ROOT = os.path.join(REPO_ROOT, "data", "eras")

MULTIVERSE_ID = "multiverse"

# The era NAME each version carries on its card. Named rather than dated,
# because the card has to read as an identity ("V Kohli - Genesis") rather than
# a date range, and three Kohlis on one auction floor need telling apart at a
# glance.
TAGS = {"2008_2013": "Genesis", "2014_2022": "The Shift", "2023_2026": "Modern Era"}

# EVERY version who took the field is emitted, and the `rateable_*` flags carried
# over from each era decide who is draftable. These are two different questions
# and conflating them costs real accuracy, exactly as ml/etl/era_players.py
# documents for the single-era pools:
#
#   the model  wants every version, so each one's balls are attributed to his own
#              row. A 150-ball cut sent 7.8% of lookups to the shared UNKNOWN row,
#              blurring hundreds of fringe versions into one average -- against
#              0.000% for every single-era table.
#   the draft  wants only versions with enough balls for a trustworthy rating,
#              which is what rateable_batting / rateable_bowling already say.
MIN_BALLS = 1
MIN_BOWL_BALLS = 1


def build() -> list:
    out = []
    for era_id, tag in TAGS.items():
        path = os.path.join(ERA_ROOT, era_id, "players.json")
        if not os.path.exists(path):
            print(f"  {era_id}: no pool -- skipped")
            continue
        with open(path, "r", encoding="utf-8") as fh:
            records = json.load(fh)

        kept = 0
        for r in records:
            bat_ok = (r.get("batting") or {}).get("balls", 0) >= MIN_BALLS
            bowl_ok = (r.get("bowling") or {}).get("legal_balls", 0) >= MIN_BOWL_BALLS
            if not (bat_ok or bowl_ok):
                continue          # never actually took the field in that era
            c = dict(r)
            c["name"] = f"{r['name']} ({tag})"
            c["real_name"] = r["name"]
            c["source_era"] = era_id
            c["era_tag"] = tag
            out.append(c)
            kept += 1
        print(f"  {era_id:<12} {kept:>4} versions kept of {len(records)}")
    return out


def main() -> None:
    print(f"building the multiverse pool (>={MIN_BALLS} balls batting or "
          f">={MIN_BOWL_BALLS} bowling)\n")
    records = build()
    d = os.path.join(ERA_ROOT, MULTIVERSE_ID)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "players.json"), "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=1)

    names = {r["real_name"] for r in records}
    multi = sum(1 for n in names
                if sum(1 for r in records if r["real_name"] == n) > 1)
    draft = sum(1 for r in records
                if r.get("rateable_batting") or r.get("rateable_bowling"))
    print(f"\n  {len(records)} entries, {len(names)} distinct cricketers")
    print(f"  {multi} appear in more than one era")
    print(f"  {draft} draftable")
    print(f"  wrote {os.path.join(d, 'players.json')}")


if __name__ == "__main__":
    main()
