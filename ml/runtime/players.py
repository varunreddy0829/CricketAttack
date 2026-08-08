"""Build engine `Batter`/`Bowler` objects from data/players_historical.json.

This mirrors `_make_batter` / `_make_bowler` / `_bat_grid` / `_bowl_grid` in
src/server.py, but standalone: those close over the server's module-level `GAME`
and `BY_NAME` globals, so importing them would drag Flask app setup into every
harness run. Reimplemented here rather than extracted, because extracting would
mean editing src/.

Tournament fatigue (`_energy_mult`) is deliberately not reproduced -- it returns
1.0 outside tournament play, which is the case for every harness innings.
"""

from __future__ import annotations

import json
import os

from src.models.player import Batter, Bowler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HISTORICAL_PATH = os.path.join(REPO_ROOT, "data", "players_historical.json")

# batting role -> which style_fit cell holds its grid (defend reads the anchor cell)
BAT_ROLE_CELL = {"attack": "attack", "rotate": "rotate", "defend": "anchor"}

ERA_ROOT = os.path.join(REPO_ROOT, "data", "eras")

# one cache per pool, keyed by era id (None = the all-time historical file)
_POOLS: dict[str | None, dict[str, dict]] = {}


def load_players(era_id: str | None = None) -> dict[str, dict]:
    """name -> raw record for one pool. Cached per era after the first call.

    `era_id=None` (or "all_time") is the original career-wide
    data/players_historical.json. Any other id reads that era's own pool, whose
    stats, playstyle grids and OVRs are all scoped to that era.
    """
    key = None if era_id in (None, "all_time") else era_id
    if key in _POOLS:
        return _POOLS[key]

    if key is None:
        with open(HISTORICAL_PATH, "r", encoding="utf-8") as fh:
            text = fh.read()
        # the file is prefixed with `//` comment lines, so it isn't valid JSON
        records = json.loads(text[text.find("["):])
    else:
        with open(os.path.join(ERA_ROOT, key, "players.json"), "r",
                  encoding="utf-8") as fh:
            records = json.load(fh)

    _POOLS[key] = {r["name"]: r for r in records}
    return _POOLS[key]


def make_batter(record: dict, intent: int = 50) -> Batter:
    b = record["batting"]
    return Batter(
        name=record["name"],
        # `or 55` covers an era pool whose OVRs haven't been derived yet -- during
        # training and during derive_ovr itself. The model path doesn't read this
        # (it runs with player_stages=False, so the classic OVR-ratio stage is
        # skipped), so a placeholder here can't affect what it predicts.
        ovr=max(1, int(record.get("batting_ovr") or 55)),
        career_runs=b["runs"],
        career_balls=b["balls"],
        fours=b["fours"],
        sixes=b["sixes"],
        dismissals=max(1, b["dismissals"]),
        intent=intent,
    )


def make_bowler(record: dict, intent: int = 50) -> Bowler:
    bw = record["bowling"]
    return Bowler(
        name=record["name"],
        ovr=max(1, int(record.get("bowling_ovr") or 55)),
        eco=bw["eco"] if bw.get("eco") and bw["eco"] > 0 else 8.5,
        wkt=bw["wickets"],
        intent=intent,
        legal_balls=bw["legal_balls"],
        style=record.get("bowling_style", "Pace"),
    )


def phase_key(over_num: int) -> str:
    """Matches src/server.py's `_phase_key` so grids are looked up identically."""
    if over_num <= 5:
        return "pp"
    if over_num <= 14:
        return "mid"
    return "death"


def bat_grid(record: dict, over_num: int, role: str) -> int:
    sf = record.get("style_fit") or {}
    cell = BAT_ROLE_CELL.get(role, "rotate")
    return (sf.get(phase_key(over_num)) or {}).get(cell, 50)


def bowl_grid(record: dict, over_num: int, role: str) -> int:
    bf = record.get("bowl_fit") or {}
    return (bf.get(phase_key(over_num)) or {}).get(role, 50)


def league_avg() -> dict:
    """The flat 25-key `league_avg` the engine stages read.

    Imported from src.server so the harness scores the engine with exactly the
    calibration the live game uses -- recomputing it here would risk drifting from
    config/baseline_weights.json. The import is read-only; Flask's app object is
    constructed but never run.
    """
    from src.server import LEAGUE_AVG
    return LEAGUE_AVG
