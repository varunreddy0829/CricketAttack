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
from ml.runtime.venues import canonical_ground
from src.utils.compile_player_stats import KNOWN_SPINNERS

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
OUT_PATH = os.path.join(ARTIFACTS, "ball_table.npz")

# league fallbacks when a venue has too little history to stand on its own
MIN_VENUE_BALLS = 600


def venue_key(raw: str) -> str:
    """One id per physical ground, shared with live play."""
    return canonical_ground(raw) or (raw or "").strip().lower().replace(" ", "_")


def _venue_aggregates(era=None):
    """-> (per-venue totals, per-(venue,match) totals, league means, character).

    `character` carries the two features that describe what KIND of ground it is
    rather than how much it scores:

      bdry_share   boundary runs / all runs. The road<->grind axis, and genuinely
                   independent of level -- Jaipur and Wankhede score at a similar
                   rate but Jaipur RUNS them (53.5% boundary share) while
                   Wankhede HITS them (62.4%).
      type_edge    the ground's economy edge for spin and for pace, RESIDUALISED:
                   each bowler is compared to his own overall economy, then those
                   deltas are averaged. Without residualising, the number mostly
                   reports who bowled there -- Chepauk reads spin-friendly on raw
                   figures purely because CSK played Jadeja and Ashwin on it.
    """
    venue = defaultdict(lambda: [0, 0, 0])          # balls, runs, wickets
    venue_match = defaultdict(lambda: [0, 0, 0])
    bdry = defaultdict(lambda: [0, 0])              # boundary runs, all runs
    tot = [0, 0, 0]
    tot_bdry = [0, 0]
    bowler_tot = defaultdict(lambda: [0, 0])        # runs, balls (whole era)
    bowler_at = defaultdict(lambda: [0, 0])         # runs, balls (at this venue)

    for innings in iter_innings():
        if era is not None and not era.covers(innings.season):
            continue
        # canonical, NOT the raw Cricsheet string: live play looks these up
        # through ml/runtime/venues.py, and "Wankhede Stadium" vs "Wankhede
        # Stadium, Mumbai" would otherwise train on half the ground's history
        # each while serving the merged figure
        v = venue_key(innings.venue)
        vm = (v, innings.match_id)
        for b in innings.balls:
            if not b.is_legal:
                continue
            for acc in (venue[v], venue_match[vm], tot):
                acc[0] += 1
                acc[1] += b.runs_total
                acc[2] += 1 if b.outcome == "Out" else 0
            br = 4 if b.outcome == "4" else 6 if b.outcome == "6" else 0
            for acc in (bdry[v], tot_bdry):
                acc[0] += br
                acc[1] += b.runs_total
            t = bowler_tot[b.bowler]; t[0] += b.runs_total; t[1] += 1
            a = bowler_at[(v, b.bowler)]; a[0] += b.runs_total; a[1] += 1

    league = (tot[1] / max(1, tot[0]), tot[2] / max(1, tot[0]))
    league_bdry = tot_bdry[0] / max(1, tot_bdry[1])

    res = defaultdict(lambda: {"spin": [0.0, 0], "pace": [0.0, 0]})
    for (v, bw), (r, bl) in bowler_at.items():
        if bl < 60:                       # too few balls for his delta to mean much
            continue
        tr, tb = bowler_tot[bw]
        if tb < 300:
            continue
        k = "spin" if bw in KNOWN_SPINNERS else "pace"
        c = res[v][k]
        c[0] += (6 * r / bl - 6 * tr / tb) * bl
        c[1] += bl

    character = {}
    for v in venue:
        d = res.get(v) or {"spin": [0.0, 0], "pace": [0.0, 0]}
        sp, pc = d["spin"], d["pace"]
        character[v] = {
            "bdry_share": (bdry[v][0] / bdry[v][1]) if bdry[v][1] else league_bdry,
            "spin": (sp[0] / sp[1]) if sp[1] >= 400 else 0.0,
            "pace": (pc[0] / pc[1]) if pc[1] >= 400 else 0.0,
        }
    return venue, venue_match, league, character, league_bdry


def build(out_path: str = OUT_PATH, era=None) -> dict:
    """Build a ball table. `era` (an ml.etl.eras.Era) scopes it to that era's
    balls AND that era's player pool -- so the model's player table only ever
    contains players who actually appeared in the era it will be used for."""
    t0 = time.time()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("[1/3] venue aggregates ...", flush=True)
    venue, venue_match, (league_rpb, league_wpb), character, league_bdry = \
        _venue_aggregates(era)
    print(f"      {len(venue)} venues; league {league_rpb:.3f} runs/ball, "
          f"{league_wpb:.4f} wkts/ball, {100*league_bdry:.1f}% boundary share")

    # MULTIVERSE: every ball is attributed to the ERA-TAGGED version of whoever
    # bowled or faced it, so Kohli's 2010 deliveries train "V Kohli (Genesis)"
    # and his 2024 ones train "V Kohli (Modern Era)".
    #
    # This is what lets the multiverse have a real model instead of a borrowed
    # one. Before it, every tagged name was a stranger to the middle era's model
    # and played off a cold start (E = A.W, no learned per-player term), so the
    # mode spanning three decades was the one read most coarsely. Tagging gives
    # each version an effect LEARNED from the balls he actually played, and puts
    # all three eras on ONE model so they are directly comparable.
    #
    # Density holds up: 290,611 balls over ~1,170 tagged identities is 248 balls
    # each, between 2023-2026's 193 and 2014-2022's 309.
    #
    # The context half is deliberately era-BLIND -- one shared set of venue and
    # phase weights across all three decades. That is the right call here: the
    # multiverse is a hypothetical league, so what should carry across is each
    # player's ability, not the run rate of the season he happened to play in.
    tag_of = None
    if era is not None and getattr(era, "is_multiverse", False):
        from ml.etl.multiverse import TAGS
        from ml.etl import eras as _E
        spans = [(_E.get(eid), t) for eid, t in TAGS.items()]

        def tag_of(name, season):          # noqa: F811
            for e2, t in spans:
                if e2.covers(season):
                    return f"{name} ({t})"
            return name

    print("[2/3] player anchors ...", flush=True)
    by_name = load_players(era.id if era else None)
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
        if era is not None and not era.covers(innings.season):
            continue
        n_innings += 1
        mi = match_ids.setdefault(innings.match_id, len(match_ids))
        uid = n_innings - 1

        # leave-one-match-out venue rates
        vkey = venue_key(innings.venue)
        vt = venue[vkey]
        vm = venue_match[(vkey, innings.match_id)]
        loo_balls = vt[0] - vm[0]
        if loo_balls >= MIN_VENUE_BALLS:
            v_rpb = (vt[1] - vm[1]) / loo_balls
            v_wpb = (vt[2] - vm[2]) / loo_balls
        else:
            v_rpb, v_wpb = league_rpb, league_wpb
        ch = character.get(vkey) or {}
        v_bdry = ch.get("bdry_share", league_bdry)
        v_edge_spin, v_edge_pace = ch.get("spin", 0.0), ch.get("pace", 0.0)

        if tag_of is None:
            nm_bat = nm_bowl = nm_ns = lambda n: n
        else:
            sn = innings.season
            nm_bat = nm_bowl = nm_ns = lambda n, _s=sn: tag_of(n, _s)

        for b in innings.balls:
            bi = name_to_idx.get(nm_bat(b.batter), 0)
            wi = name_to_idx.get(nm_bowl(b.bowler), 0)
            ni = name_to_idx.get(nm_ns(b.non_striker), 0)
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
                bowler_balls=b.bowler_balls,
                over_in_spell=b.over_in_spell,
                bat_career_balls=int(career_bat_balls[bi]),
                bowl_career_balls=int(career_bowl_balls[wi]),
                ns_sr=float(ns_sr[ni]) or 120.0,
                venue_rpb=v_rpb,
                venue_wpb=v_wpb,
                venue_bdry_share=v_bdry,
                venue_type_edge=v_edge_spin if b.bowler in KNOWN_SPINNERS else v_edge_pace,
                # every path resolves these through the SAME function -- see
                # features.player_edges for why that is not merely tidy
                edges=F.resolve_edges(by_name.get(b.batter), by_name.get(b.bowler),
                                      b.over, b.bowler in KNOWN_SPINNERS),
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--era", default=None,
                    help="build for one era id, or 'all' for every model era. "
                         "Omit for the career-wide table.")
    args = ap.parse_args()

    if args.era is None:
        build()
    else:
        from ml.etl import eras as E
        targets = E.MODEL_ERAS if args.era == "all" else [E.get(args.era)]
        for era in targets:
            print(f"\n=== {era.id}  ({era.first}-{era.last}) ===")
            out = os.path.join(ARTIFACTS, "eras", era.id, "ball_table.npz")
            build(out_path=out, era=era)
