"""THE ONLY PLACE A FEATURE ROW IS EVER BUILT.

Called by the offline ETL (ml/etl/build_table.py) and by the live adapter
(ml/runtime/adapter.py). If you need a new feature, add it to ml/etl/schema.py and
write it here -- never construct a vector anywhere else. Two code paths that build
"the same" features are the standard way a model silently rots in production.

Everything is passed as plain scalars rather than a record object, so the ETL (which
has a `BallRecord`) and the adapter (which has live server state) can both call it.
"""

from __future__ import annotations

import math

import numpy as np

from ml.etl.schema import (
    BAT_ANCHOR, BOWL_ANCHOR, CTX, N_BAT_ANCHOR, N_BOWL_ANCHOR, N_CONTEXT, N_OVERS,
)

DEATH_START = 15


def build_row(
    out: np.ndarray,
    *,
    over: int,
    ball_in_over: int,
    wickets: int,
    balls_remaining: int,
    innings_no: int,
    score: int,
    target: int | None,
    striker_balls: int,
    striker_position: int,
    partnership_balls: int,
    bowler_balls: int,
    over_in_spell: int,
    bat_career_balls: int,
    bowl_career_balls: int,
    ns_ovr: float,
    ns_sr: float,
    venue_rpb: float,
    venue_wpb: float,
) -> None:
    """Write one feature row into `out` (shape (N_CONTEXT,), pre-zeroed)."""
    o = min(max(over, 0), N_OVERS - 1)
    out[o] = 1.0

    out[CTX["ball_in_over"]] = min(ball_in_over, 6) / 6.0
    out[CTX["wickets"]] = wickets / 10.0
    out[CTX["balls_remaining"]] = balls_remaining / 120.0

    second = 1.0 if innings_no == 2 and target else 0.0
    out[CTX["is_second_innings"]] = second

    if second:
        need = max(0, target - score)
        left = max(1, balls_remaining)
        rrr = 6.0 * need / left
        out[CTX["rrr"]] = min(rrr, 30.0) / 15.0
        out[CTX["rrr_gt_8"]] = 1.0 if rrr > 8 else 0.0
        out[CTX["rrr_gt_12"]] = 1.0 if rrr > 12 else 0.0
        out[CTX["rrr_gt_16"]] = 1.0 if rrr > 16 else 0.0

    if over >= DEATH_START:
        out[CTX["wickets_x_death"]] = wickets / 10.0

    # capped on purpose -- the "getting set" effect plateaus fast and the raw
    # slope is mostly survivorship (see the note in schema.py)
    out[CTX["striker_balls"]] = min(striker_balls, 15) / 15.0
    out[CTX["is_set"]] = 1.0 if striker_balls >= 10 else 0.0
    out[CTX["striker_position"]] = min(striker_position, 11) / 11.0

    out[CTX["partnership_balls"]] = min(partnership_balls, 60) / 60.0
    out[CTX["bowler_balls"]] = min(bowler_balls, 24) / 24.0
    out[CTX["over_in_spell"]] = min(over_in_spell, 4) / 4.0

    out[CTX["bat_career_balls"]] = math.log1p(max(0, bat_career_balls)) / 10.0
    out[CTX["bowl_career_balls"]] = math.log1p(max(0, bowl_career_balls)) / 10.0

    out[CTX["nonstriker_ovr"]] = ns_ovr / 100.0
    out[CTX["nonstriker_sr"]] = min(ns_sr, 250.0) / 200.0

    out[CTX["venue_runs_per_ball"]] = venue_rpb / 2.0
    out[CTX["venue_wkts_per_ball"]] = venue_wpb * 20.0


def empty_row() -> np.ndarray:
    return np.zeros(N_CONTEXT, dtype=np.float32)


# --- player anchors --------------------------------------------------------

def bat_anchor(record: dict) -> np.ndarray:
    """f_p for a batter, from a players_historical.json record."""
    b = record.get("batting") or {}
    balls = max(1, b.get("balls", 0))
    sf = record.get("style_fit") or {}
    v = np.zeros(N_BAT_ANCHOR, dtype=np.float32)
    v[0] = math.log1p(b.get("balls", 0)) / 10.0
    v[1] = min(b.get("sr", 0.0), 250.0) / 200.0
    v[2] = min(b.get("avg", 0.0), 60.0) / 40.0
    v[3] = b.get("fours", 0) / balls
    v[4] = b.get("sixes", 0) / balls
    v[5] = b.get("dismissals", 0) / balls
    v[6] = record.get("batting_ovr", 55) / 100.0
    i = 7
    for phase in ("pp", "mid", "death"):
        cell = sf.get(phase) or {}
        for k in ("attack", "anchor", "rotate"):
            v[i] = cell.get(k, 50) / 100.0
            i += 1
    return v


def bowl_anchor(record: dict) -> np.ndarray:
    """f_p for a bowler, from a players_historical.json record."""
    bw = record.get("bowling") or {}
    balls = max(1, bw.get("legal_balls", 0))
    bf = record.get("bowl_fit") or {}
    v = np.zeros(N_BOWL_ANCHOR, dtype=np.float32)
    v[0] = math.log1p(bw.get("legal_balls", 0)) / 10.0
    v[1] = min(bw.get("eco", 8.5) or 8.5, 15.0) / 10.0
    v[2] = min(bw.get("avg", 0.0) or 30.0, 60.0) / 40.0
    v[3] = min(bw.get("sr", 0.0) or 24.0, 60.0) / 40.0
    v[4] = bw.get("wickets", 0) / balls
    v[5] = record.get("bowling_ovr", 55) / 100.0
    v[6] = 1.0 if record.get("bowling_style") == "Spin" else 0.0
    i = 7
    for phase in ("pp", "mid", "death"):
        cell = bf.get(phase) or {}
        for k in ("attack", "contain", "defend"):
            v[i] = cell.get(k, 50) / 100.0
            i += 1
    return v


def build_anchor_tables(by_name: dict) -> tuple[list[str], np.ndarray, np.ndarray]:
    """-> (names, bat_anchors (P, N_BAT_ANCHOR), bowl_anchors (P, N_BOWL_ANCHOR))

    Index 0 is reserved for UNKNOWN: an all-zero anchor and, at train time, a
    zero learned correction. That is the cold-start slot for a player the model
    has never seen.
    """
    names = ["<unknown>"] + sorted(by_name)
    bat = np.zeros((len(names), N_BAT_ANCHOR), dtype=np.float32)
    bowl = np.zeros((len(names), N_BOWL_ANCHOR), dtype=np.float32)
    for i, nm in enumerate(names[1:], start=1):
        bat[i] = bat_anchor(by_name[nm])
        bowl[i] = bowl_anchor(by_name[nm])
    return names, bat, bowl
