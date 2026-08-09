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
    STRIKER_BALL_BUCKETS,
)

DEATH_START = 15

# Shrinkage for every split-derived edge. A batter with 30 balls against spin
# will show a wild edge that is mostly noise, so each edge is pulled toward 0 by
# n/(n+K). One constant for all of them on purpose -- seven separately-tuned
# knobs would be seven things to get subtly wrong.
SPLIT_SHRINK_BALLS = 150

_EDGE_CACHE: dict[tuple[int, str], dict] = {}


def phase_of(over: int) -> str:
    """pp / mid / death. MUST match ml/etl/era_players.py::_phase -- the splits
    are bucketed there and read here, so a mismatch would silently look up the
    wrong numbers."""
    return "pp" if over <= 5 else ("mid" if over <= 14 else "death")


def _edge(part_num: float, part_den: float, whole_num: float, whole_den: float,
          scale: float) -> float:
    """(part rate - whole rate) / scale, shrunk by how much of the part we saw."""
    if part_den <= 0 or whole_den <= 0:
        return 0.0
    diff = (part_num / part_den) - (whole_num / whole_den)
    return (diff / scale) * (part_den / (part_den + SPLIT_SHRINK_BALLS))


def player_edges(record: dict) -> dict:
    """Every split-derived edge for ONE player, precomputed and shrunk.

    THE ONLY PLACE THESE ARE COMPUTED. The training ETL, the harness and the
    live server all resolve their features through resolve_edges() below, which
    calls this. That is not stylistic: `nonstriker_ovr` was pinned to a constant
    in two of those three paths and read the real rating in the third, so live
    play silently ran on a feature the model had never seen vary. Seven new
    derived features means seven new chances to repeat it, unless there is
    exactly one implementation.

    Edges are relative to the player's OWN overall rate, never the league's,
    because the anchor already carries his level. Subtracting the league instead
    would double-count quality: a batter good against everything would be
    credited twice, and a specialist flattened.
    """
    key = (id(record), record.get("name", ""))
    hit = _EDGE_CACHE.get(key)
    if hit is not None:
        return hit

    b = record.get("batting") or {}
    bw = record.get("bowling") or {}
    tot_runs, tot_balls = b.get("runs", 0), b.get("balls", 0)
    tot_outs = b.get("dismissals", 0)
    tot_bdry = b.get("fours", 0) + b.get("sixes", 0)

    out = {"vs_type": {}, "phase_bat": {}, "phase_bowl": {}}

    vs = record.get("vs_type") or {}
    for t in ("spin", "pace"):
        s = vs.get(t) or {}
        out["vs_type"][t] = (
            # /100 keeps a strike-rate edge on roughly the same scale as the
            # other normalised context features
            _edge(s.get("runs", 0), s.get("balls", 0), tot_runs, tot_balls, 1.0),
            _edge(s.get("outs", 0), s.get("balls", 0), tot_outs, tot_balls, 0.05),
            _edge(s.get("bdry", 0), s.get("balls", 0), tot_bdry, tot_balls, 0.15),
        )

    pb = record.get("phase_bat") or {}
    for ph in ("pp", "mid", "death"):
        s = pb.get(ph) or {}
        out["phase_bat"][ph] = _edge(
            s.get("runs", 0), s.get("balls", 0), tot_runs, tot_balls, 1.0)

    pw = record.get("phase_bowl") or {}
    for ph in ("pp", "mid", "death"):
        s = pw.get(ph) or {}
        out["phase_bowl"][ph] = _edge(
            s.get("runs", 0), s.get("balls", 0),
            bw.get("runs_conceded", 0), bw.get("legal_balls", 0), 1.0)

    _EDGE_CACHE[key] = out
    return out


def resolve_edges(bat_record: dict | None, bowl_record: dict | None,
                  over: int, bowler_is_spin: bool) -> dict:
    """The five player-derived context scalars for THIS exact matchup.

    Call this and pass the result straight into build_row. Do not read
    `vs_type` / `phase_bat` / `phase_bowl` anywhere else.
    """
    ph = phase_of(over)
    t = "spin" if bowler_is_spin else "pace"
    sr = out = bdry = phase_sr = phase_eco = 0.0
    if bat_record:
        e = player_edges(bat_record)
        sr, out, bdry = e["vs_type"].get(t, (0.0, 0.0, 0.0))
        phase_sr = e["phase_bat"].get(ph, 0.0)
    if bowl_record:
        phase_eco = player_edges(bowl_record)["phase_bowl"].get(ph, 0.0)
    return {
        "bat_sr_edge_vs_type": sr,
        "bat_out_edge_vs_type": out,
        "bat_bdry_edge_vs_type": bdry,
        "bat_phase_sr_edge": phase_sr,
        "bowl_phase_eco_edge": phase_eco,
    }


_BUCKET_IDX = [CTX[f"sb_{lo}_{hi}"] for lo, hi in STRIKER_BALL_BUCKETS]


def _striker_bucket(balls: int) -> int:
    for i, (lo, hi) in enumerate(STRIKER_BALL_BUCKETS):
        if lo <= balls <= hi:
            return _BUCKET_IDX[i]
    return _BUCKET_IDX[-1]


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
    bowler_balls: int,
    over_in_spell: int,
    bat_career_balls: int,
    bowl_career_balls: int,
    ns_sr: float,
    venue_rpb: float,
    venue_wpb: float,
    venue_bdry_share: float = 0.59,
    venue_type_edge: float = 0.0,
    edges: dict | None = None,
) -> None:
    """Write one feature row into `out` (shape (N_CONTEXT,), pre-zeroed).

    `edges` is the dict returned by resolve_edges() -- never hand-built.
    `balls_remaining` is still taken as an argument because the required rate is
    computed from it, but it is no longer a feature: it equals
    120 - 6*over - ball_in_over, which the 20 over one-hots plus ball_in_over
    already span exactly.
    """
    o = min(max(over, 0), N_OVERS - 1)
    out[o] = 1.0

    out[CTX["ball_in_over"]] = min(ball_in_over, 6) / 6.0
    out[CTX["wickets"]] = wickets / 10.0

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

    out[_striker_bucket(max(0, striker_balls))] = 1.0
    out[CTX["striker_position"]] = min(striker_position, 11) / 11.0

    out[CTX["bowler_balls"]] = min(bowler_balls, 24) / 24.0
    out[CTX["over_in_spell"]] = min(over_in_spell, 4) / 4.0

    out[CTX["bat_career_balls"]] = math.log1p(max(0, bat_career_balls)) / 10.0
    out[CTX["bowl_career_balls"]] = math.log1p(max(0, bowl_career_balls)) / 10.0

    out[CTX["nonstriker_sr"]] = min(ns_sr, 250.0) / 200.0

    e = edges or {}
    out[CTX["bat_sr_edge_vs_type"]] = e.get("bat_sr_edge_vs_type", 0.0)
    out[CTX["bat_out_edge_vs_type"]] = e.get("bat_out_edge_vs_type", 0.0)
    out[CTX["bat_bdry_edge_vs_type"]] = e.get("bat_bdry_edge_vs_type", 0.0)
    out[CTX["bat_phase_sr_edge"]] = e.get("bat_phase_sr_edge", 0.0)
    out[CTX["bowl_phase_eco_edge"]] = e.get("bowl_phase_eco_edge", 0.0)

    out[CTX["venue_runs_per_ball"]] = venue_rpb / 2.0
    out[CTX["venue_wkts_per_ball"]] = venue_wpb * 20.0
    out[CTX["venue_bdry_share"]] = venue_bdry_share
    out[CTX["venue_type_edge"]] = venue_type_edge


def empty_row() -> np.ndarray:
    return np.zeros(N_CONTEXT, dtype=np.float32)


# --- player anchors --------------------------------------------------------

# OVR is deliberately absent from the model's inputs -- see the note on
# BAT_ANCHOR in ml/etl/schema.py. Era OVRs are DERIVED from this model by
# ml/train/derive_ovr.py, so feeding them back in is circular; and the previous
# arrangement (pin the anchor to a constant, guard every read behind a helper)
# failed exactly where guards fail -- ml/runtime/server_ctx.py bypassed it and
# fed the live rating for `nonstriker_ovr`, so play ran on a feature the model
# had only ever seen as 55. Removing the feature removes the failure mode.


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
    i = 6
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
    v[3] = bw.get("wickets", 0) / balls
    v[4] = 1.0 if record.get("bowling_style") == "Spin" else 0.0
    i = 5
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
