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

from ml.runtime import features as F
from ml.runtime import players as P
from ml.runtime.venues import canonical_ground

OVERS_PER_INNINGS = 20

_RUNTIME_DIR = os.path.dirname(os.path.abspath(__file__))
_VENUE_STATS_PATH = os.path.join(_RUNTIME_DIR, "venue_stats.json")
_ERA_ARTIFACTS = os.path.join(os.path.dirname(_RUNTIME_DIR), "artifacts", "eras")


def _venue_path(era_id: str | None) -> str:
    if era_id in (None, "all_time"):
        return _VENUE_STATS_PATH
    return os.path.join(_ERA_ARTIFACTS, era_id, "venue_stats.json")


def _load_venue_stats(era_id: str | None = None) -> dict:
    """ground_configs.json name -> (runs/ball, wkts/ball), from real IPL history.

    Built by `python -m ml.etl.compute_venue_stats [--eras]`. Falls back to the
    league average -- both if the ground isn't recognised, and if the file is
    missing entirely, so this never hard-fails play.

    Era-scoped because a ground's character genuinely changes: Chepauk was the
    highest-scoring of the big venues in 2008-2013 (1.311 runs/ball) and among
    the lowest by 2023-2026 (1.379 against Wankhede's 1.587). Reusing one
    window's rates in another inverts it.
    """
    try:
        with open(_venue_path(era_id), "r", encoding="utf-8") as fh:
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


_VENUE_CACHE: dict[str | None, dict] = {}


def _venue_stats(era_id: str | None = None) -> dict:
    key = None if era_id in (None, "all_time") else era_id
    if key not in _VENUE_CACHE:
        _VENUE_CACHE[key] = _load_venue_stats(key)
    return _VENUE_CACHE[key]


_VENUE_STATS = _venue_stats()
LEAGUE_RPB, LEAGUE_WPB = _VENUE_STATS.get(None, (1.358, 0.0493))


def venue_rates(ground_name: str | None,
                era_id: str | None = None) -> tuple[float, float]:
    """This ground's real runs/ball and wkts/ball IN THIS ERA, or the era's league
    average if the ground isn't one ml.etl.compute_venue_stats knows."""
    stats = _venue_stats(era_id)
    key = canonical_ground(ground_name)
    return stats.get(key, stats.get(None, (LEAGUE_RPB, LEAGUE_WPB)))


_CHAR_CACHE: dict = {}
LEAGUE_BDRY_SHARE = 0.589


def _venue_character(era_id: str | None) -> dict:
    """canonical ground -> (boundary share, spin edge, pace edge), per era.

    Built by ml/etl/venue_types.py. Falls back to league-neutral, so a ground
    with no profile behaves like an average one rather than inventing character.
    """
    key = None if era_id in (None, "all_time") else era_id
    if key in _CHAR_CACHE:
        return _CHAR_CACHE[key]
    path = os.path.join(_ERA_ARTIFACTS, key, "venue_profile.json") if key else None
    out = {}
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for g, x in (data.get("grounds") or {}).items():
            se, pe = x.get("spin_edge"), x.get("pace_edge")
            out[g] = (x.get("bdry_share", LEAGUE_BDRY_SHARE),
                      se if se is not None else 0.0,
                      pe if pe is not None else 0.0)
    _CHAR_CACHE[key] = out
    return out


def venue_character(ground_name: str | None, era_id: str | None = None):
    """-> (boundary share, spin edge, pace edge) for this ground IN THIS ERA."""
    chars = _venue_character(era_id)
    key = canonical_ground(ground_name)
    return chars.get(key, (LEAGUE_BDRY_SHARE, 0.0, 0.0))


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


def enrich(ctx: dict, striker, bowler, game, *, day_sigma: float = 0.0,
           era_id: str | None = None) -> dict:
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
    v_rpb, v_wpb = venue_rates(ground_name, era_id)
    v_bdry, v_spin, v_pace = venue_character(ground_name, era_id)

    # The RAW records, from the same loader the training ETL uses, so the shared
    # edge resolver sees byte-identical inputs on both sides. Deriving these off
    # the Batter/Bowler objects instead is precisely how `nonstriker_ovr` came to
    # be pinned in training and live in play.
    pool = P.load_players(era_id)
    bat_rec = pool.get(striker.name)
    bowl_rec = pool.get(bowler.name)
    is_spin = (bowl_rec or {}).get("bowling_style") == "Spin"
    over_num = st.balls // 6

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
        "ns_sr": float(getattr(ns, "sr", 120.0)) if ns else 120.0,
        "venue_rpb": v_rpb,
        "venue_wpb": v_wpb,
        "venue_bdry_share": v_bdry,
        "venue_type_edge": v_spin if is_spin else v_pace,
        "edges": F.resolve_edges(bat_rec, bowl_rec, over_num, is_spin),
        "day_factor": _state["day_factor"],
    })
    return ctx
