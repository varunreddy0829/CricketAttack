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

_BY_NAME: dict[str, dict] | None = None


def load_players() -> dict[str, dict]:
    """name -> raw record. Cached after the first call."""
    global _BY_NAME
    if _BY_NAME is None:
        with open(HISTORICAL_PATH, "r", encoding="utf-8") as fh:
            text = fh.read()
        # the file is prefixed with `//` comment lines, so it isn't valid JSON
        records = json.loads(text[text.find("["):])
        _BY_NAME = {r["name"]: r for r in records}
    return _BY_NAME


def make_batter(record: dict, intent: int = 50) -> Batter:
    b = record["batting"]
    return Batter(
        name=record["name"],
        ovr=max(1, int(record["batting_ovr"])),
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
        ovr=max(1, int(record["bowling_ovr"])),
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
