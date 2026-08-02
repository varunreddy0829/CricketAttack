"""Innings-level statistics, computed identically for real and simulated innings.

`InningsOutcome` is the common shape. The real path builds it from the Cricsheet
replay, the simulated path builds it from the harness, and `summarize` is the only
place either is turned into numbers -- so a difference in the report is always a
difference in the cricket, never a difference in the measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

OVERS = 20


@dataclass
class InningsOutcome:
    total: int = 0
    wickets: int = 0
    legal_balls: int = 0
    lineup_size: int = 11
    chased: bool = False          # second innings that reached its target
    runs_by_over: list[int] = field(default_factory=lambda: [0] * OVERS)
    wickets_by_over: list[int] = field(default_factory=lambda: [0] * OVERS)
    balls_by_over: list[int] = field(default_factory=lambda: [0] * OVERS)
    batter_scores: list[int] = field(default_factory=list)
    counts: dict = field(default_factory=dict)   # outcome class -> n

    @property
    def all_out(self) -> bool:
        return self.wickets >= min(10, self.lineup_size - 1)


# --- real data -------------------------------------------------------------

def from_replay(innings) -> InningsOutcome:
    """Build an `InningsOutcome` from an `ml.etl.replay.InningsRecord`."""
    out = InningsOutcome(
        total=innings.total,
        wickets=innings.wickets,
        legal_balls=innings.legal_balls,
        lineup_size=max(11, len(innings.lineup)),
    )
    scores: dict[str, int] = {}
    for b in innings.balls:
        out.counts[b.outcome] = out.counts.get(b.outcome, 0) + 1
        if b.over < OVERS:
            out.runs_by_over[b.over] += b.runs_total
            if b.is_legal:
                out.balls_by_over[b.over] += 1
            if b.outcome == "Out":
                out.wickets_by_over[b.over] += 1
        if b.outcome != "wide":
            scores[b.batter] = scores.get(b.batter, 0) + b.runs_batter
        else:
            scores.setdefault(b.batter, 0)
    out.batter_scores = list(scores.values())
    return out


# --- metrics ---------------------------------------------------------------

def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = q * (len(s) - 1)
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (i - lo)


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _sd(xs) -> float:
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _over_rr_autocorr(innings: list[InningsOutcome]) -> float:
    """Lag-1 autocorrelation of per-over run rates, detrended by over index.

    Detrending matters: without it you'd just measure the powerplay-middle-death
    shape, which every innings shares, and get a large positive number that says
    nothing about momentum. Real IPL sits around 0.05-0.12 once detrended. This is
    the stopping rule for the innings latent -- land in that band and there is no
    sequence structure left worth modelling.
    """
    over_means = []
    for o in range(OVERS):
        vals = [i.runs_by_over[o] for i in innings if i.balls_by_over[o] > 0]
        over_means.append(_mean(vals))

    num = den = 0.0
    for inn in innings:
        resid = [
            inn.runs_by_over[o] - over_means[o]
            for o in range(OVERS)
            if inn.balls_by_over[o] > 0
        ]
        if len(resid) < 3:
            continue
        m = _mean(resid)
        for a, b in zip(resid, resid[1:]):
            num += (a - m) * (b - m)
        den += sum((r - m) ** 2 for r in resid)
    return num / den if den else 0.0


def summarize(innings: list[InningsOutcome]) -> dict:
    """Every acceptance-test metric, as one flat dict."""
    if not innings:
        return {}

    totals = [i.total for i in innings]
    n = len(innings)
    all_counts: dict[str, int] = {}
    for i in innings:
        for k, v in i.counts.items():
            all_counts[k] = all_counts.get(k, 0) + v

    legal = sum(all_counts.get(c, 0) for c in ("0", "1", "2", "3", "4", "6", "Out"))
    legal = legal or 1
    all_scores = [s for i in innings for s in i.batter_scores]

    # three-wicket overs: the tuning target for the wicket cascade
    three_w = sum(1 for i in innings for w in i.wickets_by_over if w >= 3)
    overs_bowled = sum(1 for i in innings for b in i.balls_by_over if b > 0) or 1

    phase_rr = {}
    for name, lo, hi in (("pp", 0, 6), ("mid", 6, 15), ("death", 15, 20)):
        runs = sum(sum(i.runs_by_over[lo:hi]) for i in innings)
        balls = sum(sum(i.balls_by_over[lo:hi]) for i in innings)
        phase_rr[name] = 6.0 * runs / balls if balls else 0.0

    return {
        "n_innings": n,
        "innings_mean": _mean(totals),
        "innings_sd": _sd(totals),
        "innings_p10": _pct(totals, 0.10),
        "innings_p50": _pct(totals, 0.50),
        "innings_p90": _pct(totals, 0.90),
        "pct_over_200": 100.0 * sum(1 for t in totals if t >= 200) / n,
        "pct_under_120": 100.0 * sum(1 for t in totals if t < 120) / n,
        "allout_rate": 100.0 * sum(1 for i in innings if i.all_out) / n,
        "wkts_per_innings": _mean(i.wickets for i in innings),
        "rr_powerplay": phase_rr["pp"],
        "rr_middle": phase_rr["mid"],
        "rr_death": phase_rr["death"],
        "dot_pct": 100.0 * all_counts.get("0", 0) / legal,
        "one_pct": 100.0 * all_counts.get("1", 0) / legal,
        "four_pct": 100.0 * all_counts.get("4", 0) / legal,
        "six_pct": 100.0 * all_counts.get("6", 0) / legal,
        "out_pct": 100.0 * all_counts.get("Out", 0) / legal,
        "wides_per_innings": all_counts.get("wide", 0) / n,
        "noballs_per_innings": all_counts.get("noball", 0) / n,
        "top_score_mean": _mean(max(i.batter_scores) for i in innings if i.batter_scores),
        "batter_score_sd": _sd(all_scores),
        "fifty_plus_per_innings": sum(1 for s in all_scores if s >= 50) / n,
        "three_wkt_overs_pct": 100.0 * three_w / overs_bowled,
        "over_rr_autocorr": _over_rr_autocorr(innings),
    }


# Metrics where being close matters, with the tolerance that counts as "matched".
# Tolerances are absolute, in the metric's own units.
ACCEPTANCE = [
    ("innings_mean", 6.0, "runs"),
    ("innings_sd", 3.0, "runs"),
    ("allout_rate", 4.0, "%"),
    ("rr_powerplay", 0.4, "rpo"),
    ("rr_middle", 0.4, "rpo"),
    ("rr_death", 0.6, "rpo"),
    ("pct_over_200", 3.0, "%"),
    ("pct_under_120", 3.0, "%"),
    ("dot_pct", 2.5, "%"),
    ("four_pct", 1.5, "%"),
    ("six_pct", 1.0, "%"),
    ("out_pct", 0.8, "%"),
    ("wkts_per_innings", 0.8, "wkts"),
    ("wides_per_innings", 1.5, "n"),
    ("three_wkt_overs_pct", 0.5, "%"),
    ("over_rr_autocorr", 0.06, ""),
]
