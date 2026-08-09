"""Longevity: reward the players who did it for years, not for one season.

The trained model reads a player's RATES -- strike rate, boundary rate, out rate.
Rates alone cannot tell a nine-season mainstay from a one-season wonder, so the
raw model ranking put T Curran (126 career runs, 3 seasons) above S Dhawan (4182
runs, 9 seasons). Both have similar rates; only one of them has proved it.

## It is a CONTEST, not two bonuses

Both players are scored, and only the DIFFERENCE is applied:

    diff = bat_score - bowl_score        (clamped to -1..+1)

That is better than adjusting each side separately for three reasons. Two proven
players cancel to the raw matchup automatically instead of by careful
construction. Two UNPROVEN players also cancel -- so a tailender is never
punished merely for being weak, only for facing someone better, which avoids
double-counting what the model's anchors already say. And it reads as cricket:
whoever is more proven wins the ball.

## The response is squared, not linear

    strength = sign(diff) x diff^2

The median matchup has |diff| = 0.48, so a linear response would apply half the
maximum adjustment to a completely ordinary contest -- the layer would never stop
talking. Squared, that same matchup gets 23%, and the layer stays quiet until
there is a real mismatch. Big gap, big change; small gap, almost nothing.

## Transfer, not a ratio

The classic engine does the same comparison by MULTIPLYING 1/2/4/6 by the OVR
ratio and letting 0 and Out absorb the rest. That is unbounded, and it breaks:
at 99 vs 55 it drives Out to exactly zero -- the batter becomes undismissable --
and a routine 90-vs-70 inflates an innings by 129%. Because Out is the innings
LENGTH multiplier, squeezing it explodes the total.

The paired-transfer rule cannot do that:

    T = dial x strength x min(total of gain buckets, total of pay buckets)

Sizing off the SMALLER side means each paying bucket loses at most `dial` of
itself, so no probability can be driven to zero however large the mismatch, and
the total is conserved exactly with no renormalising.

## Symmetric buckets, including Out

Both directions move all six buckets. An earlier version left Out alone on the
batting side to avoid compounding -- fewer dismissals means a longer innings, and
a longer innings is also being scored faster, so a 10% nudge became +32%. But
removing Out from one side only made the layer LOPSIDED (+9% batting vs -25%
bowling). Keeping the buckets symmetric and shrinking the dial fixes it properly:
at 0.10 a maximum mismatch moves an innings about +10% / -9%.

## The score, and the slider

    score = W x pctile(career index)  +  (1 - W) x pctile(quality index)

Both percentiles are taken within the player's own era, so an era with more
cricket in it doesn't inflate everyone. Centred on 0.5 and rescaled to -1..+1, so
a median player scores exactly 0.

W is the slider: 1.0 = career volume only, 0.0 = rate only. It defaults high
because rate is ALREADY fully represented in the model's own output; this layer
exists to add the thing the model cannot see.
"""

from __future__ import annotations

# Out is a SMALL bucket (~55 of 1000) and balls-survived is 1/Out, so a dial here
# multiplies hard: 0.40 stretches a maximum-mismatch innings by +70%, and 0.80 by
# +423%. It has to be far smaller than a dial that moves the big scoring buckets.
# At 0.18 a maximum mismatch adds about +22% to an innings and ~1% to strike rate,
# which is the intended shape -- more runs, same tempo.
LONGEVITY_DIAL = 0.18

# The slider. 1.0 = career volume only, 0.0 = quality only.
VOLUME_WEIGHT = 0.80

# Percentiles are measured against players with at least this many balls -- the
# YARDSTICK, not the set being scored. See build_scores for why the two must be
# separate: ranking against everyone who ever batted makes a part-timer the
# median, which flattered every mid-volume player.
REFERENCE_BALLS = 300

# The transfer is Out AGAINST EVERYTHING ELSE, not scoring-buckets against
# dots-and-Out. Being proven should mean you LAST longer, not that you suddenly
# hit harder.
#
# The earlier arrangement paid out of `0` and `Out` together, and since dots are
# six times the size of Out (320 vs 55) nearly all the mass came from dots --
# which is a strike-rate boost. At the dial needed to lift Kohli into the top ten
# it had him striking at 154 against a career 129.9, and Russell at 217.
#
# Moving Out against every other bucket in proportion leaves the SHAPE of the
# scoring distribution alone, so strike rate barely shifts (+2.3% at dial 0.40)
# while balls survived climb. More runs, at his own tempo.
BAT_GAIN, BAT_PAY = ("0", "1", "2", "3", "4", "6"), ("Out",)

_CACHE: dict[str, dict] = {}


def _percentiles(values: list[float]) -> dict[float, float]:
    """value -> fraction of the pool at or below it."""
    ranked = sorted(values)
    n = len(ranked)
    out, i = {}, 0
    for v in ranked:
        if v not in out:
            out[v] = (ranked.index(v) + sum(1 for x in ranked if x == v)) / n
    return out


def build_scores(records: list[dict], *, volume_weight: float = VOLUME_WEIGHT,
                 min_balls: int = 60,
                 reference_balls: int = REFERENCE_BALLS) -> dict:
    """name -> {'bat': score, 'bowl': score}, each in -1..+1.

    Two different pools, and keeping them apart is what makes the scores mean
    anything:

      SCORED   everyone with `min_balls`, so no one silently defaults to neutral.
      YARDSTICK only players with `reference_balls`, because the percentiles have
               to answer "how do you compare to an ESTABLISHED player".

    Ranking against everyone was the bug. With a yardstick of every player who
    ever faced 60 balls, half of it is fringe, so SS Tiwary's 536 runs scored
    +0.51 -- above the median of a pool whose median is a part-timer, and ahead
    of Kohli and Warner once the layer was applied. Measured against players with
    a real career instead, the same 536 runs score -0.03, while Warner, Kohli,
    Rahul and Dhawan barely move (+0.98/+0.87/+0.95/+0.83).

    Scores are era-relative by construction: every percentile comes from one
    era's players.
    """
    bat = [r for r in records if (r.get("batting") or {}).get("balls", 0) >= min_balls]
    bowl = [r for r in records
            if (r.get("bowling") or {}).get("legal_balls", 0) >= min_balls]
    bat_ref = [r for r in records
               if (r.get("batting") or {}).get("balls", 0) >= reference_balls]
    bowl_ref = [r for r in records
                if (r.get("bowling") or {}).get("legal_balls", 0) >= reference_balls]

    def score_side(pool, ref, career_key, quality_key):
        if not pool or not ref:
            return {}
        careers = sorted(career_key(r) for r in ref)
        quals = sorted(quality_key(r) for r in ref)
        n = len(ref)

        def pct(sorted_vals, v):
            lo = sum(1 for x in sorted_vals if x < v)
            eq = sum(1 for x in sorted_vals if x == v)
            return (lo + eq / 2.0) / n         # midrank, so ties don't jump

        out = {}
        for r in pool:
            pc = pct(careers, career_key(r))
            pq = pct(quals, quality_key(r))
            s = volume_weight * pc + (1.0 - volume_weight) * pq
            out[r["name"]] = max(-1.0, min(1.0, (s - 0.5) * 2.0))
        return out

    return {
        # career  = runs x avg x SR -- quality multiplied by how much of it there
        #           was. Chosen by measurement, not taste: scored on 2014-2018 and
        #           asked to predict 2019-2022, it ranks future runs at rho 0.501
        #           against 0.364 for avg x SR alone. Volume is the signal here --
        #           runs ALONE manages 0.495, so how much a player batted predicts
        #           his future far better than how well he batted.
        # quality = avg x SR, the standard T20 index: rewards both not getting out
        #           and scoring quickly, and is what the slider tilts toward.
        "bat": score_side(
            bat, bat_ref,
            career_key=lambda r: (r["batting"]["runs"] * r["batting"]["avg"]
                                  * r["batting"]["sr"]),
            quality_key=lambda r: r["batting"]["avg"] * r["batting"]["sr"]),
        "bowl": score_side(
            bowl, bowl_ref,
            # the bowler mirror: wickets are his volume, and a LOW average and
            # economy are his quality, so both are inverted to keep "bigger is
            # better" the meaning of every percentile here
            career_key=lambda r: (r["bowling"]["wickets"]
                                  / max(1e-9, (r["bowling"]["avg"] or 40.0)
                                        * (r["bowling"]["eco"] or 12.0))),
            quality_key=lambda r: 1.0 / max(1e-9, (r["bowling"]["avg"] or 40.0)
                                            * (r["bowling"]["eco"] or 12.0))),
    }


def scores_for(era_id: str | None, records: list[dict] | None = None,
               *, volume_weight: float = VOLUME_WEIGHT) -> dict:
    """Cached per era, since the pool never changes during a match."""
    key = f"{era_id}:{volume_weight}"
    if key not in _CACHE:
        if records is None:
            from ml.runtime.players import load_players
            records = list(load_players(era_id).values())
        _CACHE[key] = build_scores(records, volume_weight=volume_weight)
    return _CACHE[key]


def matchup_strength(bat_score: float, bowl_score: float) -> float:
    """-> -1..+1. Positive means the batter is the more proven of the two.

    Squared so ordinary matchups are left almost untouched: the median |diff| is
    0.48, which a linear response would turn into half the maximum adjustment.
    """
    diff = max(-1.0, min(1.0, bat_score - bowl_score))
    return (1.0 if diff >= 0 else -1.0) * diff * diff


def apply_longevity(weights: dict, bat_score: float = 0.0,
                    bowl_score: float = 0.0,
                    dial: float = LONGEVITY_DIAL) -> dict:
    """Tilt the model's probabilities toward whichever player is more proven.

    One transfer, sized by the DIFFERENCE between the two scores, so evenly
    matched players -- whether both great or both unknown -- get the model's raw
    output back untouched.
    """
    base = {k: float(v) for k, v in weights.items()}
    strength = matchup_strength(bat_score, bowl_score)
    if not strength:
        return base

    gain, pay = BAT_GAIN, BAT_PAY
    if strength < 0:                       # the bowler is ahead: mirror it
        gain, pay, strength = pay, gain, -strength

    g_tot = sum(base.get(k, 0.0) for k in gain)
    p_tot = sum(base.get(k, 0.0) for k in pay)
    if g_tot <= 0.0 or p_tot <= 0.0:
        return base

    # sized off the SMALLER side, so no paying bucket can lose more than `dial`
    # of itself and nothing can reach zero
    transfer = dial * strength * min(g_tot, p_tot)
    result = dict(base)
    for k in gain:
        result[k] += transfer * (base.get(k, 0.0) / g_tot)
    for k in pay:
        result[k] -= transfer * (base.get(k, 0.0) / p_tot)

    # Backstop only; the min() sizing above already makes this unreachable.
    if any(v < 0.0 for v in result.values()):
        result = {k: max(0.0, v) for k, v in result.items()}
    total = sum(result.values())
    if total > 0:
        scale = sum(base.values()) / total
        if abs(scale - 1.0) > 1e-12:
            result = {k: v * scale for k, v in result.items()}
    return result
