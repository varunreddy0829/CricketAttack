"""Per-era player pools.

    ml/.venv/Scripts/python -m ml.etl.era_players            # every model era
    ml/.venv/Scripts/python -m ml.etl.era_players --era 2023_2026

Writes data/eras/<era_id>/players.json: each player's record built ONLY from the
balls they faced or bowled inside that era, with the 3x3 playstyle grids
re-percentiled against that era's own population.

Two deliberate differences from src/utils/compile_player_stats.py:

  NO OVR.  The old compiler derived OVR from a hand-written power formula. That
      formula is the reason a 93-rated Gayle adds fewer runs than an 80-rated
      Abhishek Sharma -- it measures dominance-in-its-own-time, which does not
      convert into value against a specific era's bowling. OVR here is written
      later by ml/train/derive_ovr.py, which MEASURES each player by simulating
      them through that era's trained model. This file emits `batting_ovr: null`
      as a placeholder so the shape stays stable and a missing derivation is
      loud rather than silent.

  Grids are era-relative.  A 90th-percentile death-overs hitter in 2010 and one
      in 2025 are different players doing different things. Percentiling against
      the era's own pool is what makes the grid mean "good FOR THIS ERA", which
      is the whole point of splitting.

Everything else -- the ledger shape, the ghost-stat smoothing, the locked
volume^E * rate recipe -- is imported from the existing compiler rather than
reimplemented, so the two can't drift.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from src.utils.compile_player_stats import (
    KNOWN_FOREIGNERS,
    KNOWN_KEEPERS,
    KNOWN_SPINNERS,
    PHASES,
    _blank_bowl_phase,
    _blank_phase,
    compute_bowler_fits,
    compute_style_fits,
)
from ml.etl import eras as E
from ml.etl.replay import iter_innings

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_ROOT = os.path.join(REPO_ROOT, "data", "eras")

# batting/bowling phase buckets, matching the engine's own view of an innings
def _phase(over: int) -> str:
    return "pp" if over <= 5 else ("mid" if over <= 14 else "death")


def _blank(name: str) -> dict:
    return {
        "name": name,
        "batting": {"runs": 0, "balls": 0, "fours": 0, "sixes": 0, "dismissals": 0},
        "bowling": {"runs_conceded": 0, "legal_balls": 0, "wickets": 0},
        "bat_phase": {ph: _blank_phase() for ph in PHASES},
        "bowl_phase": {ph: _blank_bowl_phase() for ph in PHASES},
    }


def aggregate(era: E.Era) -> dict:
    """Walk this era's balls and build the per-player ledgers."""
    players: dict[str, dict] = {}
    stumped_by = defaultdict(int)

    def rec(name):
        if name not in players:
            players[name] = _blank(name)
        return players[name]

    for innings in iter_innings():
        if not era.covers(innings.season):
            continue
        for b in innings.balls:
            ph = _phase(b.over)
            bat, bowl = rec(b.batter), rec(b.bowler)

            # --- batting. A wide is not faced; everything else is.
            if b.outcome != "wide":
                d = bat["bat_phase"][ph]
                bat["batting"]["balls"] += 1
                d["balls"] += 1
                r = b.runs_batter
                bat["batting"]["runs"] += r
                d["runs"] += r
                if r == 0:
                    d["dots"] += 1
                elif r == 1:
                    d["ones"] += 1
                elif r == 2:
                    d["twos"] += 1
                elif r == 3:
                    d["threes"] += 1
                elif r == 4:
                    bat["batting"]["fours"] += 1
                    d["fours"] += 1
                elif r == 6:
                    bat["batting"]["sixes"] += 1
                    d["sixes"] += 1
                if b.outcome == "Out":
                    bat["batting"]["dismissals"] += 1
                    d["dismissals"] += 1

            # --- bowling
            wd = bowl["bowl_phase"][ph]
            bowl["bowling"]["runs_conceded"] += b.runs_total
            if b.is_legal:
                bowl["bowling"]["legal_balls"] += 1
                wd["balls"] += 1
            r = b.runs_batter
            if r in (4, 6):
                wd["br"] += r
            elif r in (1, 2, 3):
                wd["rr"] += r
            # only bowler-credited dismissals count toward a bowling grid
            if b.outcome == "Out" and b.wicket_kind != "run out":
                bowl["bowling"]["wickets"] += 1
                wd["wkts"] += 1
            if b.wicket_kind == "stumped":
                stumped_by[b.batter] += 0   # placeholder; keeper flag comes from the curated set

    return players


def finalise(players: dict, era: E.Era) -> list:
    """Derived rates, metadata, grids. Returns the records to write."""
    for p in players.values():
        b, w = p["batting"], p["bowling"]
        b["avg"] = round(b["runs"] / b["dismissals"], 2) if b["dismissals"] else float(b["runs"])
        b["sr"] = round(100.0 * b["runs"] / b["balls"], 2) if b["balls"] else 0.0
        w["eco"] = round(6.0 * w["runs_conceded"] / w["legal_balls"], 2) if w["legal_balls"] else 0.0
        w["avg"] = round(w["runs_conceded"] / w["wickets"], 2) if w["wickets"] else 0.0
        w["sr"] = round(w["legal_balls"] / w["wickets"], 2) if w["wickets"] else 0.0

    # the locked grid recipe, re-percentiled against THIS era's pool
    compute_style_fits(players)
    compute_bowler_fits(players)

    # EVERY player who appeared in the era is emitted, with `rateable_*` flags
    # marking who is good enough to draft. These are two different questions and
    # conflating them costs real accuracy:
    #
    #   the model  wants every player, so each one's balls are attributed to
    #              their own row. Dropping the fringe sent 5-7.6% of lookups to
    #              the shared UNKNOWN row, blurring ~75 players into one average.
    #   the draft  wants only players with enough balls for a trustworthy rating.
    #
    # The server filters on the flags when building the auction pool; the ETL
    # does not filter at all.
    out = []
    for name, p in sorted(players.items()):
        bat_ok = p["batting"]["balls"] >= era.min_bat_balls
        bowl_ok = p["bowling"]["legal_balls"] >= era.min_bowl_balls
        if p["batting"]["balls"] == 0 and p["bowling"]["legal_balls"] == 0:
            continue                      # never actually took the field
        out.append({
            "name": name,
            "is_keeper": name in KNOWN_KEEPERS,
            "is_foreigner": name in KNOWN_FOREIGNERS,
            "bowling_style": "Spin" if name in KNOWN_SPINNERS else "Pace",
            "role": p.get("role"),
            "roles": p.get("roles", []),
            "signature": p.get("signature", {}),
            "style_fit": p["style_fit"],
            "bowl_fit": p["bowl_fit"],
            "batting": p["batting"],
            "bowling": p["bowling"],
            # OVR is MEASURED later by ml/train/derive_ovr.py -- null here on
            # purpose so an un-derived pool fails loudly instead of shipping
            # zeros or, worse, the old formula's era-blind numbers.
            "batting_ovr": None,
            "bowling_ovr": None,
            # ...and pinned OUT of the model's inputs. derive_ovr writes the two
            # fields above from the model's own behaviour, so letting them back
            # in as features would be circular and would make a player look
            # different at play time than during training. See
            # ml/runtime/features.py::_anchor_ovr.
            "anchor_ovr": 55,
            "rateable_batting": bat_ok,
            "rateable_bowling": bowl_ok,
        })
    return out


def build(era: E.Era) -> list:
    players = aggregate(era)
    records = finalise(players, era)
    d = os.path.join(OUT_ROOT, era.id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "players.json"), "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=1)
    nb = sum(1 for r in records if r["rateable_batting"])
    nw = sum(1 for r in records if r["rateable_bowling"])
    draft = sum(1 for r in records if r["rateable_batting"] or r["rateable_bowling"])
    print(f"  {era.id:<12} {len(records):>4} in the model   "
          f"{draft:>4} draftable  ({nb} batting, {nw} bowling)")
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--era", default=None, help="one era id; default = all model eras")
    args = ap.parse_args()
    targets = [E.get(args.era)] if args.era else E.MODEL_ERAS
    print(f"building per-era player pools ({len(targets)} era(s))\n")
    for era in targets:
        build(era)


if __name__ == "__main__":
    main()
