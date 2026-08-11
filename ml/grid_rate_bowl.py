"""Rate every bowler over a real 4-OVER SPELL against a real batting order.

    ml/.venv/Scripts/python -m ml.grid_rate_bowl --era 2014_2022 --top 50

## Why a spell, and not a per-ball number

The first version of this rated a bowler by runs conceded per ball to a single
batter. That buries the one thing a strike bowler is for. Chahal took 159 wickets
in 2014-2022 -- the most in the era, at the best wicket-rate per ball of anyone
near the top -- and finished 31st, because his economy (7.71) is ordinary and a
per-ball metric is mostly economy.

A wicket's value is NOT "this batter's innings ended". It is "a WORSE batter is
now on strike for the rest of my spell". Nothing per-ball can see that, because it
never looks past the man currently facing.

So a bowler here bowls an actual spell: 24 balls, against an ordered batting
line-up, and every wicket promotes the next man in. A bowler who strikes early
spends the rest of his spell against weaker players and concedes less for it --
which means wicket-taking is priced automatically, with no invented exchange rate
between a wicket and a run.

## Exact, not simulated

The spell is a Markov chain over "how many wickets have fallen", solved forward
ball by ball:

    runs  += sum_w  P(w wickets down) x runs_per_ball(batter_w)
    wkts  += sum_w  P(w wickets down) x p_out(batter_w)
    P(w)  <- P(w) x (1 - p_out_w)  +  P(w-1) x p_out_{w-1}

Every quantity is read from the model's probabilities. Nothing is sampled, so
there is no Monte Carlo noise to average away.

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

## The same unbiased grid as the batting side

ENUMERATED   grounds x phase x batting-order strength x innings
SAMPLED      over, ball-in-over, wickets down, score, target, required rate,
             striker balls, bowler balls, non-striker SR -- drawn from REAL balls
             bowled in that era, phase and innings

Nothing is pinned to a hand-chosen value, for the reason the batting rater
documents: a pinned input silently picks a winner. The draw is seeded per cell, so
every bowler meets the identical set of situations.

Batting orders DECLINE in quality the way a real one does -- a wicket brings in
someone genuinely worse, not an equal. Three orders (strong / mid / weak) span
what a bowler actually runs into across a season.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np

from ml.etl import eras as E
from ml.etl.replay import iter_innings
from ml.etl.schema import CTX, N_CONTEXT
from ml.grid_rate import (
    PHASES, grounds, harvest_states, softmax_rows, PLAYER_COLS, LEGAL,
)
from ml.runtime import features as F
from ml.runtime import longevity as L
from ml.runtime.longevity import build_scores, matchup_strength
from ml.runtime.engine import load_calibration
from ml.runtime.model import OutcomeModel


def _art(era):
    """The era whose ARTIFACTS to read -- model, calibration, venue profile.

    Itself for a normal era; the middle era for the multiverse, which has none of
    its own and borrows them exactly as ml/runtime/engine.py does."""
    return getattr(era, "model_era", era.id) or era.id
from ml.runtime.players import load_players
from src.utils.compile_player_stats import KNOWN_SPINNERS

def is_spinner(rec) -> bool:
    """Read the record's own `bowling_style`, never the name.

    The multiverse tags names ("R Ashwin (Genesis)"), so a KNOWN_SPINNERS lookup
    misses every one of them and classifies the whole pool as pace -- which
    emptied the spin tiers and crashed the bowler picker. `bowling_style` is
    baked in per era at ETL time and survives the tagging.
    """
    return (rec.get("bowling_style") or "") == "Spin"


RUNS = np.array([0, 1, 2, 3, 4, 6, 0], dtype=np.float64)
SPELL_BALLS = 24                    # four overs
ORDER_DEPTH = 8                     # how far down the line-up a spell can reach

# Where in the batting order a spell starts, by phase. A powerplay spell meets
# the top of the order; a death spell meets whoever is left.
PHASE_START = {"pp": 0, "mid": 2, "death": 4}

# Three line-ups, as FRACTIONS of the quality-sorted batting population. Each
# DECLINES steeply, so a wicket promotes a genuinely worse player -- which is the
# entire mechanism that prices wicket-taking here.
#
# Fractions, not fixed ranks, and drawn from everyone who batted rather than the
# draftable pool. The first version used ranks into a 93-man pool and its "weak"
# order bottomed out at STR Binny (avg 15.6) -- a respectable lower-order bat, not
# a number 11. With no real tail, a wicket brought in an equal, the quality drop
# across an order measured 0.2 runs, and the derived wicket value came out at
# 0.08. A card has to end at someone who cannot bat or wickets price at nothing.
ORDERS = {
    "strong": (0.00, 0.02, 0.04, 0.08, 0.16, 0.32, 0.58, 0.82),
    "mid":    (0.10, 0.15, 0.22, 0.32, 0.46, 0.62, 0.78, 0.92),
    "weak":   (0.28, 0.38, 0.48, 0.60, 0.72, 0.83, 0.92, 0.98),
}
# Low enough to reach real tailenders, high enough that the rating is not noise.
MIN_BAT_BALLS = 40


ORDER_SHRINK = 300      # balls before a batter's own quality is half-trusted


def phase_usage(era):
    """bowler -> {phase: share of his own legal balls}.

    Weighting the three phases EQUALLY is a bias, not a neutral default. Chahal
    bowls 72% of his balls in the middle overs and 16% at the death, so a 33/33/33
    rating judges him mostly in a role he does not fill -- and at the death he
    concedes 8.77 against 7.16 in the middle. Bhuvneshwar is the mirror: 54% of
    his work is powerplay at an economy of 5.79.

    A bowler's value is what he does in the role he is actually used in, so the
    phases are weighted by his own usage. Bowlers with too few balls to have a
    stable split fall back to the league average shape.
    """
    from collections import defaultdict
    counts = defaultdict(lambda: defaultdict(int))
    league = defaultdict(int)
    for inn in iter_innings():
        if not era.covers(inn.season):
            continue
        for b in inn.balls:
            if b.is_legal:
                counts[b.bowler][b.phase] += 1
                league[b.phase] += 1
    lt = sum(league.values()) or 1
    fallback = {p: league[p] / lt for p in PHASES}
    out = {}
    for n, c in counts.items():
        t = sum(c.values())
        out[n] = fallback if t < 120 else {p: c[p] / t for p in PHASES}
    return out, fallback


def wicket_value(era) -> float:
    """What one extra wicket actually costs a batting side, in runs.

    MEASURED, not assumed: mean final score of completed innings grouped by
    wickets lost, averaged over consecutive steps. 2014-2022 gives 5.6 runs.

    The first version derived this as (runs vs a strong order - runs vs a weak
    order) / 8, which measures the gap BETWEEN line-ups rather than the cost of
    losing a wicket, and returned 0.60 -- nearly ten times too small. That single
    number is what buried the strike bowlers: Chahal takes 2.08 wickets a spell to
    Axar Patel's 1.57, and at 0.60 that edge was worth a third of a run.

    A small part of this is already counted inside the spell, since a wicket there
    promotes a worse batter for the remaining balls. The spell is 24 of 120 balls
    and the wicket falls mid-spell on average, so the overlap is roughly a tenth
    of the total -- left in rather than tuned out, since inventing a correction
    would be worse than a known small double-count.
    """
    from collections import defaultdict
    by_w = defaultdict(list)
    for inn in iter_innings():
        if not era.covers(inn.season) or not inn.balls:
            continue
        last = inn.balls[-1]
        if last.balls_remaining > 6:            # completed innings only
            continue
        w = last.wickets + (1 if last.outcome == "Out" else 0)
        by_w[w].append(last.score + last.runs_total)
    drops, prev = [], None
    for w in sorted(by_w):
        v = by_w[w]
        if len(v) < 20:
            continue
        m = sum(v) / len(v)
        if prev is not None:
            drops.append(prev - m)
        prev = m
    return sum(drops) / len(drops) if drops else 5.0


def batting_orders(records):
    """-> {tier: [record, ...]} ordered best-to-worst, like a real card.

    Sorted on avg x SR SHRUNK toward the league mean by balls faced. Raw avg x SR
    put Conway (252 balls) at slot 3 of the strong order and Powell (250) at slot
    6 -- the same small-sample flattery the ratings themselves shrink away. An
    order built on noise makes a noisy wicket value.
    """
    pool = [r for r in records if r["batting"]["balls"] >= MIN_BAT_BALLS]
    tot_b = sum(r["batting"]["balls"] for r in pool)
    league = (sum(r["batting"]["avg"] * r["batting"]["sr"] * r["batting"]["balls"]
                  for r in pool) / tot_b) if tot_b else 0.0

    def quality(r):
        """Shrink only DOWNWARD -- cap small-sample flattery, never rescue a tail.

        Symmetric shrinkage pulled Shami (64 balls, avg 6.4) up toward the mean
        and put him mid-order, which destroys the very thing this order exists
        for: a tail that genuinely cannot bat, so a wicket is worth something.
        """
        b = r["batting"]
        raw = b["avg"] * b["sr"]
        if raw <= league:
            return raw
        f = b["balls"] / (b["balls"] + ORDER_SHRINK)
        return league + (raw - league) * f

    pool.sort(key=quality, reverse=True)
    n = len(pool)
    return {t: [pool[min(n - 1, int(f * n))] for f in fracs]
            for t, fracs in ORDERS.items()}


def spell_rows(era_id, pools, phase, reps, seed):
    """Context rows for every spell in this phase, shape (n_spells, 24, F).

    One real match state is drawn per OVER of the spell and held for its six
    balls, so the over/score/wickets move the way they do in a real over rather
    than jumping every delivery.
    """
    import random
    gs = grounds(era_id)
    rows, se, pe = [], [], []
    for gname, rpb, wpb, bdry, s_edge, p_edge in gs:
        for innings in (1, 2):
            pool = pools.get((phase, innings)) or pools.get((phase, 1))
            rng = random.Random(seed * 7919 + hash((gname, innings)) % 99991)
            for _ in range(reps):
                spell = []
                for over_in_spell in range(1, 5):
                    (over, _bio, wkts, score, target, sbal, bbal,
                     _spell, ns_sr, brem) = pool[rng.randrange(len(pool))]
                    for bio in range(1, 7):
                        r = np.zeros(N_CONTEXT, dtype=np.float64)
                        F.build_row(
                            r, over=over, ball_in_over=bio, wickets=wkts,
                            balls_remaining=brem, innings_no=innings,
                            score=score, target=target, striker_balls=sbal,
                            striker_position=1, bowler_balls=(over_in_spell - 1) * 6,
                            over_in_spell=over_in_spell,
                            bat_career_balls=0, bowl_career_balls=0, ns_sr=ns_sr,
                            venue_rpb=rpb, venue_wpb=wpb,
                            venue_bdry_share=bdry, venue_type_edge=0.0,
                            edges=None)
                        spell.append(r)
                rows.append(spell)
                se.append(s_edge)
                pe.append(p_edge)
    M = np.array(rows)                       # (n_spells, 24, F)
    for c in PLAYER_COLS:
        M[:, :, CTX[c]] = 0.0
    M[:, :, CTX["striker_position"]] = 0.0   # patched per batter below
    return M, np.array(se), np.array(pe)


def rate(era, *, reps, dial, volume_weight, longevity_on, seed,
         shrink=0.0, target="zero"):
    records = list(load_players(era.id).values())
    by_name = {r["name"]: r for r in records}
    model = OutcomeModel.load(era_id=_art(era))
    if shrink:
        model.shrink_target = target
        model.set_shrinkage(records, shrink)
    cal_d = load_calibration(_art(era))
    cal, out_cal = cal_d['calibration'], cal_d['out_calibration']
    scores = build_scores(records, volume_weight=volume_weight)

    print("  harvesting real match states ...", flush=True)
    pools = harvest_states(era, by_name)
    usage, usage_fallback = phase_usage(era)
    orders = batting_orders(records)
    cands = [r for r in records if r.get("rateable_bowling")]
    ng = len(grounds(_art(era)))
    n_spells = ng * 2 * reps
    print(f"  ENUMERATED  {ng} grounds x {len(PHASES)} phases x {len(ORDERS)} "
          f"batting orders x 2 innings")
    print(f"  SAMPLED     over, ball-in-over, wickets, score, target, striker "
          f"balls,\n              bowler balls, non-striker SR -- x{reps} draws, "
          f"one per over")
    print(f"  SPELL       {SPELL_BALLS} balls vs a DECLINING order "
          f"(a wicket promotes a worse batter)")
    for t, o in orders.items():
        print(f"    {t:<7}: " + " -> ".join(r["name"] for r in o[:5]) + " -> ...")
    total = n_spells * len(PHASES) * len(ORDERS) * SPELL_BALLS * ORDER_DEPTH
    print(f"  {total:,} ball-states per bowler x {len(cands)} bowlers = "
          f"{total * len(cands):,} evaluations\n")

    li = [model.ci[c] for c in LEGAL]
    phase_over = {"pp": 2, "mid": 10, "death": 17}
    Bcol = {c: model.B[CTX[c]] for c in PLAYER_COLS}
    pos_col = model.B[CTX["striker_position"]]
    venue_col = model.B[CTX["venue_type_edge"]]

    blocks = {}
    for ph in PHASES:
        M, se, pe = spell_rows(_art(era), pools, ph, reps, seed)
        blocks[ph] = (se, pe, M @ model.B)     # (n_spells, 24, 9)

    # per-batter constants, cached once
    bat_info = {}
    for t, order in orders.items():
        for slot, r in enumerate(order):
            n = r["name"]
            if n not in bat_info:
                bat_info[n] = (
                    model.effect(n, "bat", r) @ model.V_bat,
                    scores["bat"].get(n, L.FLOOR),
                    np.log1p(max(0, r["batting"]["balls"])) / 10.0,
                    r)

    out = []
    t0 = time.time()
    for i, rec in enumerate(cands, 1):
        is_spin = is_spinner(rec)
        bowl_eff = model.effect(rec["name"], "bowl", rec) @ model.V_bowl
        ws = scores["bowl"].get(rec["name"], L.FLOOR)
        bowl_cb = np.log1p(max(0, rec["bowling"]["legal_balls"])) / 10.0

        agg = defaultdict(lambda: np.zeros(2))
        n_cell = defaultdict(float)
        w_ph = usage.get(rec["name"], usage_fallback)
        for ph in PHASES:
            se, pe, Z0 = blocks[ph]
            edge = (se if is_spin else pe)[:, None, None] * venue_col
            start = PHASE_START[ph]
            for tier, order in orders.items():
                # probabilities for every batter who could face this spell
                pout, rpb = [], []
                for slot in range(ORDER_DEPTH):
                    brec = order[min(slot + start, len(order) - 1)]
                    bat_eff, bs, bat_cb, braw = bat_info[brec["name"]]
                    e = F.resolve_edges(braw, rec, phase_over[ph], is_spin)
                    delta = (bat_cb * Bcol["bat_career_balls"]
                             + bowl_cb * Bcol["bowl_career_balls"]
                             + (min(slot + start + 1, 11) / 11.0) * pos_col)
                    for k in ("bat_sr_edge_vs_type", "bat_out_edge_vs_type",
                              "bat_bdry_edge_vs_type", "bat_phase_sr_edge",
                              "bowl_phase_eco_edge"):
                        delta = delta + e.get(k, 0.0) * Bcol[k]
                    Z = Z0 + edge + (model.alpha + delta + bat_eff + bowl_eff)
                    sh = Z.shape
                    P = softmax_rows(Z.reshape(-1, sh[2])).reshape(sh)[:, :, li]
                    if cal != 1.0 or out_cal != 1.0:
                        P[:, :, 1:6] *= cal
                        P[:, :, 6] *= out_cal
                    P = P / P.sum(axis=2, keepdims=True)
                    p_o = P[:, :, 6]
                    if longevity_on:
                        s = matchup_strength(bs, ws)
                        f = dial * abs(s) * (L.PENALTY_SCALE if s < 0 else 1.0)
                        t_ = f * p_o
                        rest = 1.0 - p_o
                        if s > 0:
                            P[:, :, :6] *= (1.0 + t_ / rest)[:, :, None]
                            p_o = p_o - t_
                        elif s < 0:
                            P[:, :, :6] *= (1.0 - t_ / rest)[:, :, None]
                            p_o = p_o + t_
                    pout.append(p_o)
                    rpb.append(P[:, :, :6] @ RUNS[:6])

                # forward the wicket chain across the whole spell
                ns = pout[0].shape[0]
                Pw = np.zeros((ns, ORDER_DEPTH))
                Pw[:, 0] = 1.0
                runs = np.zeros(ns)
                wkts = np.zeros(ns)
                for t_i in range(SPELL_BALLS):
                    po = np.stack([p[:, t_i] for p in pout], axis=1)
                    rb = np.stack([r[:, t_i] for r in rpb], axis=1)
                    runs += (Pw * rb).sum(axis=1)
                    wkts += (Pw * po).sum(axis=1)
                    survive = Pw * (1.0 - po)
                    fall = Pw * po
                    Pw = survive
                    Pw[:, 1:] += fall[:, :-1]
                    Pw[:, -1] += fall[:, -1]     # absorb at the last slot
                cell = np.array([runs.mean(), wkts.mean()])
                agg[("phase", ph)] += cell
                n_cell[("phase", ph)] += 1
                agg[("tier", tier)] += cell * w_ph[ph]
                n_cell[("tier", tier)] += w_ph[ph]
                agg["all"] += cell * w_ph[ph]
                n_cell["all"] += w_ph[ph]
        b = rec["bowling"]
        a = agg["all"] / n_cell["all"]
        out.append({
            "name": rec["name"], "runs": a[0], "wkts": a[1],
            "style": "spin" if is_spin else "pace",
            **{f"{ph}_r": agg[("phase", ph)][0] / n_cell[("phase", ph)] for ph in PHASES},
            **{f"{ph}_w": agg[("phase", ph)][1] / n_cell[("phase", ph)] for ph in PHASES},
            **{f"{t}_r": agg[("tier", t)][0] / n_cell[("tier", t)] for t in ORDERS},
            "lscore": ws, "balls": b["legal_balls"], "cwkts": b["wickets"],
            "use_pp": w_ph["pp"], "use_mid": w_ph["mid"], "use_dth": w_ph["death"],
            "eco": b["eco"], "avg": b["avg"],
        })
        if i % 25 == 0:
            print(f"    {i}/{len(cands)}  ({time.time() - t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--era", default="2014_2022")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--dial", type=float, default=L.LONGEVITY_DIAL)
    ap.add_argument("--w", type=float, default=L.VOLUME_WEIGHT)
    ap.add_argument("--shrink", type=float, default=500.0)
    ap.add_argument("--target", default="zero", choices=("anchor", "mean", "zero", "replacement"))
    ap.add_argument("--no-longevity", action="store_true")
    ap.add_argument("--wicket-value", type=float, default=None,
                    help="runs a wicket costs the batting side; default is "
                         "measured from real completed innings in this era")
    args = ap.parse_args()

    era = E.get(args.era)
    print(f"\n{era.id}  {era.label}")
    print("  steps A + B only -- no calibration, roles, gambits or pitch layer")
    print(f"  shrink K={args.shrink} toward {args.target}, longevity dial {args.dial}\n")
    rows = rate(era, reps=args.reps, dial=args.dial, volume_weight=args.w,
                longevity_on=not args.no_longevity, seed=args.seed,
                shrink=args.shrink, target=args.target)

    # A wicket also costs the batting side AFTER the spell ends. Measured from
    # real completed innings rather than derived from the grid.
    wv = args.wicket_value if args.wicket_value is not None else wicket_value(era)
    for r in rows:
        r["impact"] = r["runs"] - wv * r["wkts"]
    rows.sort(key=lambda r: r["impact"])

    print(f"\n  TOP {args.top} BOWLERS -- 4-over spell vs a real batting order")
    print(f"  impact = runs conceded - {wv:.2f} x wickets "
          f"(wicket value derived from the order's own quality drop)\n")
    print(f"  {'#':>3} {'bowler':<21}{'impact':>8}{'runs':>7}{'wkts':>6}{'style':>6}"
          f"{'balls':>6}{'W':>5}{'eco':>6} |{'pp':>6}{'mid':>6}{'dth':>6}"
          f" |{'vs strong':>10}{'vs weak':>8} |{'usage pp/mid/dth':>18}")
    print("  " + "-" * 108)
    for i, r in enumerate(rows[:args.top], 1):
        print(f"  {i:>3} {r['name']:<21}{r['impact']:>8.2f}{r['runs']:>7.2f}"
              f"{r['wkts']:>6.2f}{r['style']:>6}{r['balls']:>6}{r['cwkts']:>5}"
              f"{r['eco']:>6.2f} |{r['pp_r']:>6.1f}{r['mid_r']:>6.1f}{r['death_r']:>6.1f}"
              f" |{r['strong_r']:>10.1f}{r['weak_r']:>8.1f}"
              f" |{100*r['use_pp']:>6.0f}{100*r['use_mid']:>6.0f}{100*r['use_dth']:>6.0f}")


if __name__ == "__main__":
    main()
