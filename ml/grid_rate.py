"""Rate every batter by ENUMERATING situations, not by simulating matches.

    ml/.venv/Scripts/python -m ml.grid_rate --era 2014_2022 --reps 100 --top 40

## Why not a simulation

ml/rate_batters.py plays 250 innings per player and reads the scorecard. Two
holes come with that:

  COVERAGE  The situations a player meets are whatever the dice deal him. A
            number 7 almost never bats in the powerplay, so his powerplay
            ability is inferred from innings he never played.
  NOISE     Every ball is a die-roll, so the rating is a Monte Carlo ESTIMATE of
            a number the model can state exactly.

This module states the number exactly, over a situation set built on purpose.

## Enumerate what discriminates, SAMPLE the rest -- from reality

Four axes are enumerated, because they are what a batter should be compared
across and reported on:

    grounds x phase x batting position x bowler x innings

Everything else -- over, ball in over, wickets down, score, target and required
rate, balls the striker has faced, balls the bowler has sent down, his over in
the spell, the non-striker's strike rate -- is SAMPLED FROM REAL BALLS bowled in
that era, in that phase, in that innings.

Sampling beats pinning, and this is the whole reason the file was rewritten.
Pinning `bowler_balls = 6` and `ns_sr = 130` and "over 2, 10 and 17" is not
neutral: it rates every player inside one arbitrary situation chosen by whoever
wrote the file. Measured on the pinned version, 31 of the model's 54 context
features NEVER FIRED -- 17 of the 20 over one-hots, all 5 chase features, and 9
of the 11 striker-balls buckets. A third of what the model knows was unreachable.

Drawing from real balls removes the arbitrary choice without inventing anything:
the joint distribution IS cricket, so no cell contains an opener facing his first
ball with eight wickets down.

## Random across cells, IDENTICAL across players

The draw is seeded by the cell index. Cell 4,127 holds one specific match state,
and all 163 batters are measured in it. That keeps the sampling unbiased while
keeping the comparison paired -- one player can never draw an easier set than
another.

## What it measures

Per cell the model gives 9 probabilities, and two summaries fall out:

    runs per ball    = sum(p x runs) over the legal outcomes
    balls survived   = 1 / p(Out)

Neither works alone. Runs per ball ignores getting out, so it ranks the biggest
hitters. Runs per DISMISSAL is the T20 trap -- it treats wickets as the scarce
resource when in a 20-over game the scarce resource is BALLS, so it puts an
accumulator above a man who scores twice as fast and is out one ball sooner.

So the rating is EXPECTED RUNS FROM A FIXED BALL ALLOWANCE: give every batter the
same WINDOW balls and ask what he actually scores, given he may be out first.

    E[runs] = runs_per_ball x (1 - (1 - p_out) ** WINDOW) / p_out

Exact, not sampled. It pays for surviving AND for scoring quickly, and stops
paying for survival once a player already outlasts the window -- which is what
makes it a T20 measure rather than a first-class one.

## Calibration IS applied -- roles, gambits and the pitch layer are not

Those are different kinds of thing. Roles and gambits are TACTICAL choices a
player makes during a match, so including them would rate a decision rather than
a cricketer. Calibration is not a tactic: it is the pair of constants that make
the engine produce real cricket, and leaving it out rates an engine nobody plays.

Measured, it is not a small difference. Without calibration the rater produced
10.8 wickets and 129 runs per innings against a real 5.86 and 161 -- a 1.85x
wicket rate. Since a wicket is then credited at its REAL price (5.57 runs), that
inflated count over-paid strike bowlers, who are mostly pace: the marquee tier
went from spin 15 / pace 14 under the old OVRs to spin 4 / pace 18, even though
this era's spinners are the more economical group (7.70 against 8.62) and give up
only 0.15 wickets per 24 balls.

## Steps A and B only

No calibration, roles, gambit cards or pitch layer. The ground still matters,
because venue rates are INPUTS to step A; what is excluded is the separate pitch
modifier applied after it.

## Why it is fast

The model is linear in the feature row:

    logits = alpha + row @ B + E_bat @ V_bat + E_bowl @ V_bowl

so the context rows can be built ONCE per phase and multiplied by B once. Within
a (batter, bowler, phase) block the seven player-dependent columns are constant,
which makes their whole contribution a single 9-vector added to every row. What
would be 132 million scalar model calls becomes a few thousand matrix operations.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict

import numpy as np

from ml.etl import eras as E
from ml.etl.replay import iter_innings
from ml.etl.schema import CTX, N_CONTEXT
from ml.runtime import features as F
from ml.runtime import longevity as L
from ml.runtime.longevity import build_scores, matchup_strength
from ml.runtime.engine import load_calibration
from ml.runtime.model import OutcomeModel
from ml.runtime.players import load_players
from src.utils.compile_player_stats import KNOWN_SPINNERS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGAL = ("0", "1", "2", "3", "4", "6", "Out")
RUNS = np.array([0, 1, 2, 3, 4, 6, 0], dtype=np.float64)

# The ball allowance a batter is judged over is SAMPLED per cell from how many
# balls players at that position really faced in this era -- never pinned.
#
# Pinning it was the same mistake as pinning bowler_balls: a fixed 30 sits at the
# 84th percentile of reality (real median 11, mean 15.5) and it silently picks a
# winner, because a long allowance rewards survivors and a short one rewards
# hitters. Measured across 20/30/45/60/90, Livingstone moved from 6th to 48th and
# Dhoni from 25th to 5th purely on that choice.
#
# It also varies enormously by position -- an opener faces 22.7 balls on average,
# a number 7 faces 9.2 -- so the pool is per position.
WINDOW_FALLBACK = 20

PHASES = ("pp", "mid", "death")
POSITIONS = {"pp": (1, 2, 3), "mid": (3, 4, 5), "death": (5, 6, 7)}

# A tier is only meaningful if the bowlers in it are front-line bowlers. At a
# 10-wicket bar the "elite" slots went to cameos off a few hundred balls, which
# rates batters against a bowler nobody would pick.
MIN_BOWL_BALLS = 600
PER_TIER = 5                       # 5 spinners + 5 pacers in each of 3 tiers
BOWLER_TIERS = ("elite", "mid", "weak")

# The seven feature columns that depend on WHO is playing. Everything else in the
# row is pure context and is baked into the per-phase base matrix.
PLAYER_COLS = ("bat_career_balls", "bowl_career_balls",
               "bat_sr_edge_vs_type", "bat_out_edge_vs_type",
               "bat_bdry_edge_vs_type", "bat_phase_sr_edge",
               "bowl_phase_eco_edge")


# --------------------------------------------------------------------------
# real match states


def harvest_states(era: E.Era, by_name: dict) -> dict:
    """(phase, innings) -> list of real match states from that era's own balls.

    Every field the model reads that is not enumerated below comes from here, so
    the situations are joined the way cricket joins them rather than the way a
    nested loop would.
    """
    pools = defaultdict(list)
    for inn in iter_innings():
        if not era.covers(inn.season):
            continue
        for b in inn.balls:
            ns = by_name.get(b.non_striker)
            pools[(b.phase, b.innings_no)].append((
                b.over, b.ball_in_over, b.wickets, b.score, b.target,
                b.striker_balls, b.bowler_balls, b.over_in_spell,
                float((ns or {}).get("batting", {}).get("sr", 130.0) or 130.0),
                b.balls_remaining,
            ))
    return pools


def harvest_windows(era: E.Era) -> dict:
    """position -> every real "balls faced in an innings" for that position."""
    by_pos = defaultdict(list)
    for inn in iter_innings():
        if not era.covers(inn.season):
            continue
        faced, pos = defaultdict(int), {}
        for b in inn.balls:
            if b.outcome != "wide":
                faced[b.batter] += 1
                pos.setdefault(b.batter, b.striker_position)
        for n, f in faced.items():
            by_pos[pos[n]].append(f)
    return by_pos


def pick_bowlers(records: list[dict]) -> list[tuple]:
    """30 real bowlers: 5 spinners and 5 pacers in each of elite / mid / weak.

    Real players, so both the bowler's model effect and his longevity score are
    genuine. Split on bowling average within style, then five evenly spaced picks
    from each third.
    """
    out = []
    for style in ("spin", "pace"):
        pool = [r for r in records
                if r.get("rateable_bowling")
                and r["bowling"]["legal_balls"] >= MIN_BOWL_BALLS
                and ((r["name"] in KNOWN_SPINNERS) == (style == "spin"))]
        pool.sort(key=lambda r: r["bowling"]["avg"])
        third = max(1, len(pool) // 3)
        for t, tier in enumerate(BOWLER_TIERS):
            band = pool[t * third:(t + 1) * third] or pool[-1:]
            for k in range(PER_TIER):
                out.append((tier, style, band[min(len(band) - 1,
                                                  k * len(band) // PER_TIER)]))
    return out


MIN_BAT_BALLS = 600
BAT_TIERS = ("elite", "mid", "weak")
PER_BAT_TIER = 10                  # 30 opposition batters, 10 per tier


def pick_batters(records: list[dict]) -> list[tuple]:
    """30 real batters spanning elite / mid / weak, as the opposition a bowler
    is measured against.

    Split on avg x SR -- the standard T20 index -- within the pool that faced
    enough balls to have a trustworthy one. Real players, so both their model
    effect and their longevity score are genuine.
    """
    pool = [r for r in records
            if r.get("rateable_batting")
            and r["batting"]["balls"] >= MIN_BAT_BALLS]
    pool.sort(key=lambda r: r["batting"]["avg"] * r["batting"]["sr"], reverse=True)
    third = max(1, len(pool) // 3)
    out = []
    for t, tier in enumerate(BAT_TIERS):
        band = pool[t * third:(t + 1) * third] or pool[-1:]
        for k in range(PER_BAT_TIER):
            out.append((tier, band[min(len(band) - 1,
                                       k * len(band) // PER_BAT_TIER)]))
    return out


def grounds(era_id: str) -> list[tuple]:
    path = os.path.join(REPO, "ml", "artifacts", "eras", era_id, "venue_profile.json")
    with open(path, "r", encoding="utf-8") as fh:
        g = json.load(fh)["grounds"]
    return [(k, v["rpb"], v["wpb"], v["bdry_share"],
             v.get("spin_edge") or 0.0, v.get("pace_edge") or 0.0)
            for k, v in sorted(g.items())]


# --------------------------------------------------------------------------
# the grid


def base_rows(era_id: str, pools: dict, phase: str, reps: int, seed: int,
              wpools: dict):
    """Context rows for one phase: grounds x positions x innings x reps.

    The seven player columns are left at zero and patched later. `is_spin` is
    returned per row because the venue edge depends on the bowler's style, which
    is not known until the bowler loop -- so two variants of that one column are
    built and selected from.
    """
    gs = grounds(era_id)
    rows, spin_edge, pace_edge, inn_of, windows = [], [], [], [], []
    n = 0
    for gname, rpb, wpb, bdry, se, pe in gs:
        for pos in POSITIONS[phase]:
            for innings in (1, 2):
                pool = pools.get((phase, innings)) or pools.get((phase, 1))
                rng = random.Random(seed * 1_000_003 + hash((gname, pos, innings)) % 99991)
                for _ in range(reps):
                    (over, bio, wkts, score, target, sbal, bbal,
                     spell, ns_sr, brem) = pool[rng.randrange(len(pool))]
                    r = np.zeros(N_CONTEXT, dtype=np.float64)
                    F.build_row(
                        r, over=over, ball_in_over=bio, wickets=wkts,
                        balls_remaining=brem, innings_no=innings, score=score,
                        target=target, striker_balls=sbal, striker_position=pos,
                        bowler_balls=bbal, over_in_spell=spell,
                        bat_career_balls=0, bowl_career_balls=0, ns_sr=ns_sr,
                        venue_rpb=rpb, venue_wpb=wpb, venue_bdry_share=bdry,
                        venue_type_edge=0.0, edges=None)
                    rows.append(r)
                    spin_edge.append(se)
                    pace_edge.append(pe)
                    inn_of.append(innings)
                    wp = wpools.get(pos) or [WINDOW_FALLBACK]
                    windows.append(wp[rng.randrange(len(wp))])
                    n += 1
    M = np.array(rows)
    for c in PLAYER_COLS:
        M[:, CTX[c]] = 0.0
    return (M, np.array(spin_edge), np.array(pace_edge), np.array(inn_of),
            np.maximum(1, np.array(windows, dtype=np.float64)))


def softmax_rows(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def evaluate(P, strength, dial, window, cal=1.0, out_cal=1.0):
    """P: (n, 7) legal probabilities. Applies step B, returns E[runs] per row.

    The transfer is the same paired rule as apply_longevity, written for whole
    columns at once: Out is always the smaller side, so min(gain, pay) is p_out
    and the whole thing reduces to two scale factors.
    """
    if cal != 1.0 or out_cal != 1.0:
        P = P.copy()
        P[:, 1:6] *= cal            # 1/2/3/4/6 -- the scoring buckets
        P[:, 6] *= out_cal
    P = P / P.sum(axis=1, keepdims=True)
    p_out = P[:, 6].copy()
    if strength:
        f = dial * abs(strength)
        if strength < 0:
            f *= L.PENALTY_SCALE
        t = f * p_out                       # transfer, sized off the smaller side
        rest = 1.0 - p_out
        if strength > 0:
            P[:, :6] *= (1.0 + t / rest)[:, None]
            p_out = p_out - t
        else:
            P[:, :6] *= (1.0 - t / rest)[:, None]
            p_out = p_out + t
        P[:, 6] = p_out
    rpb = P[:, :6] @ RUNS[:6]
    p_out = np.clip(p_out, 1e-9, 1.0)
    return rpb * (1.0 - (1.0 - p_out) ** window) / p_out, rpb, 1.0 / p_out


def rate(era: E.Era, *, reps: int, dial: float, volume_weight: float,
         longevity_on: bool, seed: int, contrast: float = 1.0,
         shrink: float = 0.0, target: str = 'anchor'):
    records = list(load_players(era.id).values())
    by_name = {r["name"]: r for r in records}
    model = OutcomeModel.load(era_id=era.id)
    if shrink:
        model.shrink_target = target
        model.set_shrinkage(records, shrink)
    cal_d = load_calibration(era.id)
    cal, out_cal = cal_d['calibration'], cal_d['out_calibration']
    scores = build_scores(records, volume_weight=volume_weight)
    if contrast != 1.0:
        for side in scores:
            scores[side] = {n: (1.0 if v >= 0 else -1.0) * abs(v) ** contrast
                            for n, v in scores[side].items()}

    t0 = time.time()
    print("  harvesting real match states ...", flush=True)
    pools = harvest_states(era, by_name)
    wpools = harvest_windows(era)
    tot_states = sum(len(v) for v in pools.values())
    print(f"    {tot_states:,} real balls, "
          + ", ".join(f"{p}/{i}={len(pools[(p, i)]):,}"
                      for p in PHASES for i in (1, 2) if (p, i) in pools))

    bowlers = pick_bowlers(records)
    cands = [r for r in records if r.get("rateable_batting")]
    ng = len(grounds(era.id))
    per_phase = ng * 3 * 2 * reps
    print(f"  ENUMERATED  {ng} grounds x {len(PHASES)} phases x 3 positions "
          f"x {len(bowlers)} bowlers x 2 innings")
    print(f"  SAMPLED     over, ball-in-over, wickets, score, target, striker "
          f"balls,\n              bowler balls, over-in-spell, non-striker SR "
          f"-- x{reps} draws")
    print(f"    windows sampled per position: "
          + ", ".join(f"{k}:{sum(v)/len(v):.0f}" for k, v in sorted(wpools.items())[:8]))
    print(f"  {per_phase * len(PHASES) * len(bowlers):,} cells per batter "
          f"x {len(cands)} batters = "
          f"{per_phase * len(PHASES) * len(bowlers) * len(cands):,} evaluations")

    li = [model.ci[c] for c in LEGAL]

    # per-phase context, multiplied through B once
    phase_over = {"pp": 2, "mid": 10, "death": 17}
    blocks = {}
    for ph in PHASES:
        M, se, pe, inn, win = base_rows(era.id, pools, ph, reps, seed, wpools)
        blocks[ph] = (M, se, pe, M @ model.B, inn == 1, inn == 2, win)
    print(f"  context built in {time.time() - t0:.0f}s", flush=True)

    Bcol = {c: model.B[CTX[c]] for c in PLAYER_COLS}
    venue_col = model.B[CTX["venue_type_edge"]]

    brecs = []
    for tier, style, r in bowlers:
        eff = model.effect(r["name"], "bowl", r) @ model.V_bowl
        brecs.append((tier, style, r, eff,
                      scores["bowl"].get(r["name"], L.FLOOR)))

    out = []
    t0 = time.time()
    for bi, rec in enumerate(cands, 1):
        bat_eff = model.effect(rec["name"], "bat", rec) @ model.V_bat
        bs = scores["bat"].get(rec["name"], L.FLOOR)
        bat_cb = np.log1p(max(0, rec["batting"]["balls"])) / 10.0

        tot = np.zeros(4)          # expected, rpb, balls, n
        by_phase = defaultdict(float); n_phase = defaultdict(int)
        by_tier = defaultdict(float); n_tier = defaultdict(int)
        by_inn = defaultdict(float); n_inn = defaultdict(int)
        for ph in PHASES:
            M, se, pe, Z0, m1, m2, win = blocks[ph]
            for tier, style, brec, beff, ws in brecs:
                is_spin = style == "spin"
                e = F.resolve_edges(rec, brec, phase_over[ph], is_spin)
                delta = (bat_cb * Bcol["bat_career_balls"]
                         + (np.log1p(brec["bowling"]["legal_balls"]) / 10.0)
                         * Bcol["bowl_career_balls"])
                for k in ("bat_sr_edge_vs_type", "bat_out_edge_vs_type",
                          "bat_bdry_edge_vs_type", "bat_phase_sr_edge",
                          "bowl_phase_eco_edge"):
                    delta = delta + e.get(k, 0.0) * Bcol[k]
                Z = Z0 + (se if is_spin else pe)[:, None] * venue_col
                Z = Z + (model.alpha + delta + bat_eff + beff)
                P = softmax_rows(Z)[:, li]
                strength = matchup_strength(bs, ws) if longevity_on else 0.0
                ev, rpb, balls = evaluate(P, strength, dial, win, cal, out_cal)
                m = ev.mean()
                tot += [ev.sum(), rpb.sum(), np.minimum(balls, 1e6).sum(), len(ev)]
                by_phase[ph] += m; n_phase[ph] += 1
                by_tier[tier] += m; n_tier[tier] += 1
                by_inn[1] += ev[m1].mean(); n_inn[1] += 1
                by_inn[2] += ev[m2].mean(); n_inn[2] += 1
        n = tot[3]
        out.append({
            "name": rec["name"],
            "expected": tot[0] / n, "rpb": tot[1] / n, "balls": tot[2] / n,
            "pp": by_phase["pp"] / n_phase["pp"],
            "mid": by_phase["mid"] / n_phase["mid"],
            "death": by_phase["death"] / n_phase["death"],
            "vs_elite": by_tier["elite"] / n_tier["elite"],
            "vs_mid": by_tier["mid"] / n_tier["mid"],
            "vs_weak": by_tier["weak"] / n_tier["weak"],
            "inn1": by_inn[1] / n_inn[1],
            "inn2": by_inn[2] / n_inn[2],
            "lscore": bs,
            "real_runs": rec["batting"]["runs"],
            "real_sr": rec["batting"]["sr"],
        })
        if bi % 25 == 0:
            print(f"    {bi}/{len(cands)}  ({time.time() - t0:.0f}s)", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--era", default="2014_2022")
    ap.add_argument("--reps", type=int, default=100,
                    help="draws of the sampled dimensions per enumerated combo")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--dial", type=float, default=L.LONGEVITY_DIAL)
    ap.add_argument("--w", type=float, default=L.VOLUME_WEIGHT)
    ap.add_argument("--no-longevity", action="store_true")
    ap.add_argument("--target", default="anchor",
                    choices=("anchor", "mean", "zero", "replacement"),
                    help="what shrinkage pulls a player toward")
    ap.add_argument("--shrink", type=float, default=0.0,
                    help="pull a batter's model effect toward league "
                         "average by balls/(balls+K); 0 = off")
    ap.add_argument("--contrast", type=float, default=1.0,
                    help="stretch the longevity score: sign(s)*|s|**k. "
                         "1.0 = as-is; higher widens the gap between a long "
                         "career and a short one without changing the order.")
    args = ap.parse_args()

    era = E.get(args.era)
    print(f"\n{era.id}  {era.label}")
    print("  steps A + B only -- no calibration, roles, gambits or pitch layer")
    print(f"  longevity: "
          + ("OFF" if args.no_longevity else f"dial {args.dial}, W {args.w}"))
    print("  rating: expected runs from a ball allowance sampled per position "
          "from real innings")
    print(f"  score contrast k = {args.contrast}\n")

    rows = rate(era, reps=args.reps, dial=args.dial, volume_weight=args.w,
                longevity_on=not args.no_longevity, seed=args.seed,
                contrast=args.contrast, shrink=args.shrink,
                target=args.target)
    rows.sort(key=lambda r: -r["expected"])

    print(f"\n  TOP {args.top} BATTERS")
    print(f"  {'#':>3}  {'batter':<21}{'E[runs]':>9}{'r/ball':>8}{'balls':>7}"
          f" |{'pp':>7}{'mid':>7}{'death':>7} |{'elite':>7}{'weak':>7}"
          f" |{'bat 1st':>8}{'chase':>7}"
          f" |{'L':>6}{'runs':>7}")
    print("  " + "-" * 110)
    for i, r in enumerate(rows[:args.top], 1):
        print(f"  {i:>3}  {r['name']:<21}{r['expected']:>9.1f}{r['rpb']:>8.2f}"
              f"{r['balls']:>7.1f} |{r['pp']:>7.1f}{r['mid']:>7.1f}{r['death']:>7.1f}"
              f" |{r['vs_elite']:>7.1f}{r['vs_weak']:>7.1f}"
              f" |{r['inn1']:>8.1f}{r['inn2']:>7.1f}"
              f" |{r['lscore']:>+6.2f}{r['real_runs']:>7}")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# the bowling side


def rate_bowlers(era: E.Era, *, reps: int, dial: float, volume_weight: float,
                 longevity_on: bool, seed: int, shrink: float = 0.0,
                 target: str = "anchor"):
    """Rate every bowler by how few runs the opposition batter is expected to
    score against him.

    Deliberately the SAME metric as the batting side rather than an economy or a
    wicket rate. `E[runs]` already folds dismissal into scoring -- a bowler who
    takes the wicket ends the batter's allowance, and one who dries up the runs
    lowers it a different way -- so a single number weighs the two correctly
    without needing an invented exchange rate between a wicket and a run.

    Lower is better, so the table is sorted ascending.
    """
    records = list(load_players(era.id).values())
    by_name = {r["name"]: r for r in records}
    model = OutcomeModel.load(era_id=era.id)
    if shrink:
        model.shrink_target = target
        model.set_shrinkage(records, shrink)
    scores = build_scores(records, volume_weight=volume_weight)

    print("  harvesting real match states ...", flush=True)
    pools = harvest_states(era, by_name)
    wpools = harvest_windows(era)
    batters = pick_batters(records)
    cands = [r for r in records if r.get("rateable_bowling")]
    ng = len(grounds(era.id))
    per_phase = ng * 3 * 2 * reps
    print(f"  ENUMERATED  {ng} grounds x {len(PHASES)} phases x 3 positions "
          f"x {len(batters)} batters x 2 innings")
    print(f"  {per_phase * len(PHASES) * len(batters):,} cells per bowler "
          f"x {len(cands)} bowlers = "
          f"{per_phase * len(PHASES) * len(batters) * len(cands):,} evaluations\n")

    li = [model.ci[c] for c in LEGAL]
    phase_over = {"pp": 2, "mid": 10, "death": 17}
    blocks = {}
    for ph in PHASES:
        M, se, pe, inn, win = base_rows(era.id, pools, ph, reps, seed, wpools)
        blocks[ph] = (M, se, pe, M @ model.B, win)

    Bcol = {c: model.B[CTX[c]] for c in PLAYER_COLS}
    venue_col = model.B[CTX["venue_type_edge"]]

    brs = []
    for tier, r in batters:
        brs.append((tier, r,
                    model.effect(r["name"], "bat", r) @ model.V_bat,
                    scores["bat"].get(r["name"], L.FLOOR),
                    np.log1p(max(0, r["batting"]["balls"])) / 10.0))

    out = []
    t0 = time.time()
    for i, rec in enumerate(cands, 1):
        is_spin = rec["name"] in KNOWN_SPINNERS
        bowl_eff = model.effect(rec["name"], "bowl", rec) @ model.V_bowl
        ws = scores["bowl"].get(rec["name"], L.FLOOR)
        bowl_cb = np.log1p(max(0, rec["bowling"]["legal_balls"])) / 10.0

        tot = np.zeros(2)
        by_phase = defaultdict(float); n_phase = defaultdict(int)
        by_tier = defaultdict(float); n_tier = defaultdict(int)
        for ph in PHASES:
            M, se, pe, Z0, win = blocks[ph]
            for tier, brec, bat_eff, bs, bat_cb in brs:
                e = F.resolve_edges(brec, rec, phase_over[ph], is_spin)
                delta = bat_cb * Bcol["bat_career_balls"] + bowl_cb * Bcol["bowl_career_balls"]
                for k in ("bat_sr_edge_vs_type", "bat_out_edge_vs_type",
                          "bat_bdry_edge_vs_type", "bat_phase_sr_edge",
                          "bowl_phase_eco_edge"):
                    delta = delta + e.get(k, 0.0) * Bcol[k]
                Z = Z0 + (se if is_spin else pe)[:, None] * venue_col
                Z = Z + (model.alpha + delta + bat_eff + bowl_eff)
                P = softmax_rows(Z)[:, li]
                strength = matchup_strength(bs, ws) if longevity_on else 0.0
                ev, rpb, balls = evaluate(P, strength, dial, win, cal, out_cal)
                m = ev.mean()
                tot += [ev.sum(), len(ev)]
                by_phase[ph] += m; n_phase[ph] += 1
                by_tier[tier] += m; n_tier[tier] += 1
        b = rec["bowling"]
        out.append({
            "name": rec["name"],
            "conceded": tot[0] / tot[1],
            "style": "spin" if is_spin else "pace",
            "pp": by_phase["pp"] / n_phase["pp"],
            "mid": by_phase["mid"] / n_phase["mid"],
            "death": by_phase["death"] / n_phase["death"],
            "vs_elite": by_tier["elite"] / n_tier["elite"],
            "vs_weak": by_tier["weak"] / n_tier["weak"],
            "lscore": ws, "balls": b["legal_balls"], "wkts": b["wickets"],
            "eco": b["eco"], "avg": b["avg"],
        })
        if i % 25 == 0:
            print(f"    {i}/{len(cands)}  ({time.time() - t0:.0f}s)", flush=True)
    return out
