"""Replay -> ml/artifacts/ball_table.npz, the training matrix.

    python -m ml.etl.build_table

Two passes over the replay (it costs ~2s, so this is cheaper than holding 290k
records in memory): the first accumulates venue aggregates, the second emits rows
with leave-one-match-out venue features. LOO matters because otherwise a venue's
"historical" scoring rate includes the very match being predicted.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict

import numpy as np

from ml.etl.replay import CLASS_INDEX, iter_innings
from ml.etl.schema import N_CONTEXT, SCHEMA_HASH
from ml.runtime import features as F
from ml.runtime.players import load_players

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
OUT_PATH = os.path.join(ARTIFACTS, "ball_table.npz")

# league fallbacks when a venue has too little history to stand on its own
MIN_VENUE_BALLS = 600


def _venue_aggregates():
    """-> (per-venue totals, per-(venue,match) totals, league means)."""
    venue = defaultdict(lambda: [0, 0, 0])          # balls, runs, wickets
    venue_match = defaultdict(lambda: [0, 0, 0])
    tot = [0, 0, 0]

    for innings in iter_innings():
        v = innings.venue
        vm = (v, innings.match_id)
        for b in innings.balls:
            if not b.is_legal:
                continue
            for acc in (venue[v], venue_match[vm], tot):
                acc[0] += 1
                acc[1] += b.runs_total
                acc[2] += 1 if b.outcome == "Out" else 0

    league = (tot[1] / max(1, tot[0]), tot[2] / max(1, tot[0]))
    return venue, venue_match, league


def build(out_path: str = OUT_PATH) -> dict:
    t0 = time.time()
    os.makedirs(ARTIFACTS, exist_ok=True)

    print("[1/3] venue aggregates ...", flush=True)
    venue, venue_match, (league_rpb, league_wpb) = _venue_aggregates()
    print(f"      {len(venue)} venues; league {league_rpb:.3f} runs/ball, "
          f"{league_wpb:.4f} wkts/ball")

    print("[2/3] player anchors ...", flush=True)
    by_name = load_players()
    names, bat_anchors, bowl_anchors = F.build_anchor_tables(by_name)
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    # career volumes, used as context features (when to hedge)
    career_bat_balls = np.zeros(len(names), dtype=np.int32)
    career_bowl_balls = np.zeros(len(names), dtype=np.int32)
    ns_ovr = np.zeros(len(names), dtype=np.float32)
    ns_sr = np.zeros(len(names), dtype=np.float32)
    for nm, i in name_to_idx.items():
        r = by_name.get(nm)
        if not r:
            continue
        career_bat_balls[i] = (r.get("batting") or {}).get("balls", 0)
        career_bowl_balls[i] = (r.get("bowling") or {}).get("legal_balls", 0)
        ns_ovr[i] = r.get("batting_ovr", 55)
        ns_sr[i] = (r.get("batting") or {}).get("sr", 0.0)
    print(f"      {len(names) - 1} players (+1 unknown slot)")

    print("[3/3] emitting rows ...", flush=True)
    X_list, y_list = [], []
    bat_i, bowl_i, ns_i = [], [], []
    season_l, match_l, innings_l, legal_l = [], [], [], []
    match_ids: dict[str, int] = {}
    unknown = 0
    n_innings = 0

    for innings in iter_innings():
        n_innings += 1
        mi = match_ids.setdefault(innings.match_id, len(match_ids))
        uid = n_innings - 1

        # leave-one-match-out venue rates
        vt = venue[innings.venue]
        vm = venue_match[(innings.venue, innings.match_id)]
        loo_balls = vt[0] - vm[0]
        if loo_balls >= MIN_VENUE_BALLS:
            v_rpb = (vt[1] - vm[1]) / loo_balls
            v_wpb = (vt[2] - vm[2]) / loo_balls
        else:
            v_rpb, v_wpb = league_rpb, league_wpb

        for b in innings.balls:
            bi = name_to_idx.get(b.batter, 0)
            wi = name_to_idx.get(b.bowler, 0)
            ni = name_to_idx.get(b.non_striker, 0)
            unknown += (bi == 0) + (wi == 0)

            row = np.zeros(N_CONTEXT, dtype=np.float32)
            F.build_row(
                row,
                over=b.over,
                ball_in_over=b.ball_in_over,
                wickets=b.wickets,
                balls_remaining=b.balls_remaining,
                innings_no=b.innings_no,
                score=b.score,
                target=b.target,
                striker_balls=b.striker_balls,
                striker_position=b.striker_position,
                partnership_balls=b.partnership_balls,
                bowler_balls=b.bowler_balls,
                over_in_spell=b.over_in_spell,
                bat_career_balls=int(career_bat_balls[bi]),
                bowl_career_balls=int(career_bowl_balls[wi]),
                ns_ovr=float(ns_ovr[ni]) or 55.0,
                ns_sr=float(ns_sr[ni]) or 120.0,
                venue_rpb=v_rpb,
                venue_wpb=v_wpb,
            )
            X_list.append(row)
            y_list.append(CLASS_INDEX[b.outcome])
            bat_i.append(bi)
            bowl_i.append(wi)
            ns_i.append(ni)
            season_l.append(b.season)
            match_l.append(mi)
            innings_l.append(uid)
            legal_l.append(1 if b.is_legal else 0)

    X = np.asarray(X_list, dtype=np.float32)
    payload = dict(
        X=X,
        y=np.asarray(y_list, dtype=np.int8),
        bat_idx=np.asarray(bat_i, dtype=np.int32),
        bowl_idx=np.asarray(bowl_i, dtype=np.int32),
        ns_idx=np.asarray(ns_i, dtype=np.int32),
        season=np.asarray(season_l, dtype=np.int16),
        match_idx=np.asarray(match_l, dtype=np.int32),
        innings_uid=np.asarray(innings_l, dtype=np.int32),
        is_legal=np.asarray(legal_l, dtype=np.int8),
        bat_anchors=bat_anchors,
        bowl_anchors=bowl_anchors,
        player_names=np.asarray(names),
        schema_hash=np.asarray(SCHEMA_HASH),
    )
    np.savez_compressed(out_path, **payload)

    print(f"\n      {X.shape[0]} balls x {X.shape[1]} features, "
          f"{n_innings} innings, {len(match_ids)} matches")
    print(f"      {unknown} unmatched player references "
          f"({100 * unknown / max(1, 2 * X.shape[0]):.3f}% of lookups)")
    print(f"      seasons {min(season_l)}-{max(season_l)}")
    print(f"      wrote {out_path} "
          f"({os.path.getsize(out_path) / 1e6:.1f} MB) in {time.time() - t0:.1f}s")
    print(f"      schema {SCHEMA_HASH}")
    return payload


if __name__ == "__main__":
    build()
