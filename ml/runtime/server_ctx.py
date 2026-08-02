"""Read live match state off `src.server.GAME` and enrich the engine's ctx.

The server builds a 10-key ctx at src/server.py:1013 that carries no score, no
wickets, no target and no balls-faced -- the model needs all of it. Rather than
edit the server to plumb them through, this reads them from the module global.

That is deliberately experiment scaffolding, not a design. It keeps `src/`
byte-identical while the two versions are being compared. If the model is adopted,
the real change is extending that ctx dict at the source.
"""

from __future__ import annotations

import json
import os
import random

from ml.runtime.venues import canonical_ground

OVERS_PER_INNINGS = 20

_VENUE_STATS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "venue_stats.json")


def _load_venue_stats() -> dict:
    """ground_configs.json name -> (runs/ball, wkts/ball), from real IPL history.

    Built by `python -m ml.etl.compute_venue_stats`. Falls back to the league
    average -- both if the ground isn't recognised, and if the file is missing
    entirely (e.g. before it's been generated), so this never hard-fails play.
    """
    try:
        with open(_VENUE_STATS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except OSError:
        return {}
    fb = data.get("_league_fallback", {"runs_per_ball": 1.358, "wkts_per_ball": 0.0493})
    out = {None: (fb["runs_per_ball"], fb["wkts_per_ball"])}
    for key, entry in data.items():
        if key.startswith("_"):    # "_league_fallback", "_season_min" -- metadata, not a ground
            continue
        out[key] = (entry["runs_per_ball"], entry["wkts_per_ball"])
    return out


_VENUE_STATS = _load_venue_stats()
LEAGUE_RPB, LEAGUE_WPB = _VENUE_STATS.get(None, (1.358, 0.0493))


def venue_rates(ground_name: str | None) -> tuple[float, float]:
    """The CURRENT match's ground's real runs/ball and wkts/ball, or the league
    average if the ground isn't one of the ones ml.etl.compute_venue_stats knows."""
    key = canonical_ground(ground_name)
    return _VENUE_STATS.get(key, _VENUE_STATS.get(None, (LEAGUE_RPB, LEAGUE_WPB)))


_state: dict = {"innings": None, "day_factor": 0.0, "pship_key": None,
                "pship_balls": 0, "spell_last_over": {}, "spell_len": {}}


def reset() -> None:
    _state.update({"innings": None, "day_factor": 0.0, "pship_key": None,
                   "pship_balls": 0, "spell_last_over": {}, "spell_len": {}})


def _over_in_spell(bowler_name: str, over: int) -> int:
    """Consecutive overs in the bowler's CURRENT unbroken spell.

    Must match ml/etl/replay.py exactly. Bowlers alternate ends, so consecutive
    overs by one bowler are two apart; any other gap starts a fresh spell.

    An earlier version returned total overs bowled this innings -- a different
    quantity. A bowler in his 3rd over of the innings but 1st of a new spell read
    as 3 where training said 1, so the model applied what it knew about a bowler
    deep into a spell to a fresh one.
    """
    last = _state["spell_last_over"].get(bowler_name)
    if last == over:
        return _state["spell_len"].get(bowler_name, 1)   # another ball in the same over
    if last == over - 2:
        _state["spell_len"][bowler_name] = _state["spell_len"].get(bowler_name, 0) + 1
    else:
        _state["spell_len"][bowler_name] = 1
    _state["spell_last_over"][bowler_name] = over
    return _state["spell_len"][bowler_name]


def enrich(ctx: dict, striker, bowler, game, *, day_sigma: float = 0.0) -> dict:
    """Return `ctx` with the model's match-state keys added. Never mutates `game`."""
    st = game.get("state")
    if st is None:
        return ctx

    innings = game.get("innings", 1)
    # the day factor is drawn ONCE per innings and held -- persistent randomness is
    # the only kind that moves innings-total variance
    if _state["innings"] != innings:
        _state["innings"] = innings
        _state["day_factor"] = random.gauss(0.0, day_sigma) if day_sigma else 0.0
        _state["pship_key"] = None
        _state["pship_balls"] = 0
        _state["spell_last_over"] = {}      # spells don't carry across an innings
        _state["spell_len"] = {}

    # the server tracks no partnership counter; derive one from (wickets, innings)
    key = (innings, st.wickets)
    if _state["pship_key"] != key:
        _state["pship_key"] = key
        _state["pship_balls"] = 0
    else:
        _state["pship_balls"] += 1

    bat_row = (game.get("bat_card") or {}).get(striker.name) or {}
    bowl_row = (game.get("bowl_card") or {}).get(bowler.name) or {}
    ns = st.get_non_striker() if hasattr(st, "get_non_striker") else None

    ground_name = (game.get("match_ground") or {}).get("name")
    v_rpb, v_wpb = venue_rates(ground_name)

    ctx = dict(ctx)
    ctx.update({
        "ball_in_over": (st.balls % 6) + 1,
        "score": st.runs,
        "wickets": st.wickets,
        "balls_remaining": max(0, OVERS_PER_INNINGS * 6 - st.balls),
        "innings_no": innings,
        "target": game.get("target"),
        "striker_balls": bat_row.get("balls", 0),
        "striker_position": (st.striker_index or 0) + 1,
        "partnership_balls": _state["pship_balls"],
        "bowler_balls": bowl_row.get("balls", 0),
        "over_in_spell": _over_in_spell(bowler.name, st.balls // 6),
        "bat_career_balls": getattr(striker, "career_balls", 0),
        "bowl_career_balls": getattr(bowler, "legal_balls", 0),
        "ns_ovr": float(getattr(ns, "ovr", 55)) if ns else 55.0,
        "ns_sr": float(getattr(ns, "sr", 120.0)) if ns else 120.0,
        "venue_rpb": v_rpb,
        "venue_wpb": v_wpb,
        "day_factor": _state["day_factor"],
    })
    return ctx
