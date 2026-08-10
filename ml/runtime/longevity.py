"""Longevity: reward the players who did it for years, not for one season.

The trained model reads a player's RATES -- strike rate, boundary rate, out rate.
Rates cannot tell a nine-season mainstay from someone who scored 100 off 50 balls
once: both look identical to it, so the fringe player plays like the star. This
layer sits after the model and shifts its probabilities by how much a player
actually DID in that era.

## It is a CONTEST, not two bonuses

Both players are scored, and only the DIFFERENCE is applied:

    gap = bat_score - bowl_score

Applying the two scores separately instead was measured and rejected. It fails to
cancel: Kohli (+0.94) against Bumrah (+0.89) should hand back the raw model, and
one shot does exactly that (18.2 balls survived, unchanged), while two steps give
20.1 -- and two ordinary players drift from 18.2 to 19.3. Every neutral matchup
gets quietly longer, league-wide. Two steps also move about twice as far for the
same ranking, so they cost twice the realism budget. (Order of application makes
no difference; that was checked separately.)

## The response is a straight line

    strength = gap / 2

Scores live in -1..+1, so the gap lives in -2..+2, and dividing by 2 turns it into
a fraction of the largest gap possible: 0 means no change, +1 means the most a
batter can ever be favoured, -1 the most a bowler can. Nothing is thrown away.

Two things this deliberately is NOT:

  It is not clamped. An earlier version clamped the gap at 1.0, which destroyed
  exactly the differences this layer exists to express -- 0.94, 1.92 and the
  best-vs-worst 2.00 all collapsed onto the same value. Do not reintroduce a clamp
  anywhere in this file.

  It is not squared. A previous version squared it to keep ordinary matchups
  quiet, but that was an assumption bolted on top of the design rather than part
  of it, and it made the curve arbitrary -- a matchup 57% of the way to the
  maximum got 32% of the effect for no stated reason. Straight-line means
  LONGEVITY_DIAL alone decides how strong longevity feels.

Do NOT instead let the gap run unbounded and clip the result at zero. At gap 1.91
an unbounded strength of 3.65 gives a transfer of 11.05 against an `Out` of 10.1,
so clipping makes the batter literally undismissable -- and the gain side still
receives 11.05, leaving the total at 974.8 instead of 1000. Rescaling the input is
safe; clipping the output is not.

## Transfer, not a ratio

The classic engine does the same comparison by MULTIPLYING 1/2/4/6 by the OVR
ratio and letting 0 and Out absorb the rest. That is unbounded, and it breaks: at
99 vs 55 it drives Out to exactly zero, and a routine 90-vs-70 inflates an innings
by 129%. Because Out is the innings LENGTH multiplier, squeezing it explodes the
total.

The paired-transfer rule cannot do that:

    T = dial x strength x min(total of gain buckets, total of pay buckets)

Sizing off the SMALLER side means each paying bucket loses at most `dial` of
itself, so no probability can be driven negative however large the mismatch, and
the total is conserved exactly with no renormalising. This is a proof, not a
guard: a paying bucket loses `T x base[k]/pay_total <= dial x strength x base[k]`,
which is `<= base[k]` whenever `dial <= 1` and `strength <= 1`. There is therefore
no clamp on the output -- a dial above 1 raises ValueError instead.

## The score

    score = 0.8 x percentile(volume) + 0.2 x percentile(quality)

  batting   volume = runs.  quality = avg x SR.
  bowling   volume = wickets x legal balls (output x workload).
            quality = 1 / (avg x strike rate) -- both are lower-is-better, so the
            reciprocal keeps "bigger is better" true of every number in this file.

Rate is capped at 20% deliberately, because the model ALREADY knows it. Same over,
same bowler, same score, only the batter changing, the model returns SR 181.6 for
Russell, 155.2 for de Villiers, 132.2 for Kohli and 130.4 for Dhawan -- each within
a few points of the real career figure. A rate-heavy index therefore counts the
same evidence twice: `runs x avg x SR` (and `runs^2 x avg^2 x SR`, which ranks
within 0.02 of it) let a 536-run player outrank on average alone.

Everyone is ranked against EVERYONE WHO PLAYED, and anyone who never batted (or
never bowled) takes the floor rather than defaulting to average. Before that rule,
Buttler -- who has not bowled a ball in this era -- scored 0.00 and so was rated
far above Raina, who actually bowled 318 of them for 4 wickets.
"""

from __future__ import annotations

# Out is a SMALL bucket (~10-55 of 1000) and balls-survived is 1/Out, so a dial
# here multiplies hard. It must be far smaller than a dial that moves the big
# scoring buckets.
#
# The dial IS the strength of the whole layer, and it reads directly: it is the
# largest fraction of `Out` that can ever be removed. The safe range is exactly
# [0, 1], because `strength` maxes at 1.0 and `dial x strength` must stay <= 1 --
# at 1.0 the most extreme matchup drives Out to zero and the batter becomes
# undismissable, so 1.0 is a wall, not a setting.
#
# At 0.5, on a normal distribution (Out = 55 per 1000):
#
#   gap +2 (the most a batter can be favoured)   Out -50%,  innings 2.00x
#   gap +1.97 (a proven batter vs a never-bowler) Out -49%,  innings 1.97x
#   gap -2 (the most a bowler can be favoured)   Out +25%,  innings 0.80x
#
# The two sides differ because PENALTY_SCALE halves the losing side, and because
# `Out` can always be added to but can only ever give up what it has.
LONGEVITY_DIAL = 0.5

# The slider: 1.0 = volume only, 0.0 = quality only. Rate is already fully
# represented in the model's own output, so this stays high.
VOLUME_WEIGHT = 0.80

# Losing the contest costs half of what winning it pays. A fringe player should be
# nudged, not punished -- the model already knows he is worse.
PENALTY_SCALE = 0.5

# What a player who never batted, or never bowled, scores. NOT 0.0: that is the
# median, and it rated a man who has never bowled above a genuine part-timer.
FLOOR = -1.0

# The yardstick the percentiles are ranked against -- "all" (everyone who took
# part) or "draftable". See build_scores for what each does to the spread.
REFERENCE_POOL = "all"

# The transfer is Out AGAINST EVERYTHING ELSE, not scoring-buckets against
# dots-and-Out. Being proven should mean you LAST longer, not that you suddenly
# hit harder, so every other bucket gains the same PERCENTAGE and the shape of the
# scoring distribution is untouched.
#
# Steering the transferred mass into 4s and 6s instead was measured on Kohli vs
# Raina: 3.0 units spread proportionally gives 210.3 runs per innings, and forced
# entirely into boundaries gives 211.6. One run out of a 64-run gain -- the
# innings-length multiplier dominates so completely that the destination barely
# registers. Proportional is therefore free, and it keeps the "same tempo"
# guarantee exact.
BAT_GAIN, BAT_PAY = ("0", "1", "2", "3", "4", "6"), ("Out",)

_CACHE: dict[str, dict] = {}


def build_scores(records: list[dict], *,
                 volume_weight: float = VOLUME_WEIGHT,
                 pool: str = REFERENCE_POOL) -> dict:
    """name -> {'bat': score, 'bowl': score}, each in -1..+1.

    `pool` picks the YARDSTICK the percentiles are taken against:

      "all"        every player who took part -- 380 batters, 303 bowlers.
      "draftable"  only those good enough to be drafted -- 163 and 144.

    The choice matters more than it looks, because over half the wider pool faced
    fewer than 100 balls, so its median is 71 runs against the draftable pool's
    549. Measured:

                          all      draftable
        V Kohli  4194   +0.97        +0.94
        SS Tiwary 536   +0.63        +0.15
        SK Raina        -0.03        -0.97
        100 runs        -0.10        -0.98

    Either is defensible -- "all" is the more literal reading of how much a player
    did -- but "all" compresses the top, and combined with the squared response it
    leaves too little between a 4194-run career and a 536-run one to act on.
    """
    if pool not in ("all", "draftable"):
        raise ValueError(f"pool must be 'all' or 'draftable', got {pool!r}")

    def score_side(vol_key, qual_key, has_played, drafted):
        keep = has_played if pool == "all" else (
            lambda r: has_played(r) and drafted(r))
        ref = [r for r in records if keep(r)]
        if not ref:
            return {}
        vols = sorted(vol_key(r) for r in ref)
        quals = sorted(qual_key(r) for r in ref)
        n = len(ref)

        def pct(sorted_vals, v):
            lo = sum(1 for x in sorted_vals if x < v)
            eq = sum(1 for x in sorted_vals if x == v)
            return (lo + eq / 2.0) / n          # midrank, so ties don't jump

        out = {}
        for r in records:
            if not has_played(r):
                out[r["name"]] = FLOOR           # never did it at all
                continue
            s = (volume_weight * pct(vols, vol_key(r))
                 + (1.0 - volume_weight) * pct(quals, qual_key(r)))
            # (s - 0.5) * 2 needs no clamp: s is a blend of two percentiles and is
            # therefore already in [0, 1] by construction.
            out[r["name"]] = (s - 0.5) * 2.0
        return out

    def bowl_quality(r):
        b = r["bowling"]
        # a wicketless bowler stores avg = sr = 0.0; treat him as the worst rather
        # than dividing by zero
        if not b["wickets"] or not b["avg"] or not b["sr"]:
            return 0.0
        return 1.0 / (b["avg"] * b["sr"])

    return {
        "bat": score_side(
            vol_key=lambda r: r["batting"]["runs"],
            qual_key=lambda r: r["batting"]["avg"] * r["batting"]["sr"],
            has_played=lambda r: r["batting"]["balls"] > 0,
            drafted=lambda r: bool(r.get("rateable_batting"))),
        "bowl": score_side(
            vol_key=lambda r: r["bowling"]["wickets"] * r["bowling"]["legal_balls"],
            qual_key=bowl_quality,
            has_played=lambda r: r["bowling"]["legal_balls"] > 0,
            drafted=lambda r: bool(r.get("rateable_bowling"))),
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
    """-> -1..+1, the gap expressed as a fraction of the largest gap possible.

    Straight line, deliberately:

        gap  0  ->  0.0   nothing happens
        gap +2  ->  1.0   the most a batter can ever be favoured
        gap -2  -> -1.0   the most a bowler can ever be favoured

    Scores span -1..+1, so the gap spans -2..+2 and dividing by 2 is a change of
    units, not a judgement -- it turns "gap" into "fraction of the maximum".

    An earlier version squared this, to keep ordinary matchups quiet. That was an
    extra assumption on top of the design rather than part of it, and it made the
    curve hard to reason about: a matchup 57% of the way to the maximum received
    32% of the effect for no stated reason. Linear means the one number left to
    choose -- LONGEVITY_DIAL -- fully determines how strong longevity feels, which
    is the only knob that should exist.
    """
    return (bat_score - bowl_score) / 2.0


def apply_longevity(weights: dict, bat_score: float = 0.0,
                    bowl_score: float = 0.0,
                    dial: float = LONGEVITY_DIAL,
                    penalty_scale: float = PENALTY_SCALE) -> dict:
    """Tilt the model's probabilities toward whichever player is more proven.

    One transfer, sized by the DIFFERENCE between the two scores, so evenly
    matched players -- whether both great or both unknown -- get the model's raw
    output back untouched.
    """
    if dial > 1.0:
        raise ValueError(
            f"longevity dial {dial} exceeds 1.0; the transfer's non-negativity "
            f"proof requires dial <= 1. Lower it rather than clipping the result.")

    base = {k: float(v) for k, v in weights.items()}
    strength = matchup_strength(bat_score, bowl_score)
    if not strength:
        return base

    gain, pay = BAT_GAIN, BAT_PAY
    if strength < 0:                       # the bowler is ahead: mirror it
        gain, pay, strength = pay, gain, -strength
        dial *= penalty_scale              # losing costs less than winning pays

    g_tot = sum(base.get(k, 0.0) for k in gain)
    p_tot = sum(base.get(k, 0.0) for k in pay)
    if g_tot <= 0.0 or p_tot <= 0.0:
        return base

    # Sized off the SMALLER side, so each paying bucket loses at most `dial` of
    # itself. With dial <= 1 and strength <= 1 that is provably non-negative, so
    # nothing below needs a floor and the total is conserved without renormalising.
    transfer = dial * strength * min(g_tot, p_tot)
    result = dict(base)
    for k in gain:
        result[k] += transfer * (base.get(k, 0.0) / g_tot)
    for k in pay:
        result[k] -= transfer * (base.get(k, 0.0) / p_tot)
    return result
