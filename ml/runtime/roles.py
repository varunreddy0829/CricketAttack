"""Role play: the layer that turns the model's "gold probs" into player control.

The model says what a matchup naturally produces. This layer lets the two humans
tilt that -- at a cost. It is the ONLY place a player's choice affects a ball, so
it is what makes the game a game rather than a simulation you watch.

## The one rule

Every role names buckets that GAIN and buckets that PAY:

    T = dial x min(total of gain buckets, total of pay buckets)
    each gain bucket  +=  T x (its share of the gain side)
    each pay  bucket  -=  T x (its share of the pay side)

Sizing off the SMALLER side is what keeps it safe. Each paying bucket can lose at
most `dial` of itself, so a probability can never be driven negative -- not for a
tailender with a 1% boundary chance, not for anyone. Total is conserved exactly,
so the result is still a valid distribution with no renormalising needed.

In practice the boundary buckets (4/6/Out ~13%) are much smaller than the safe
buckets (0/1 ~78%), so the rare side moves by the full dial while the common side
barely notices. Attack raises boundaries 30%; dots give up only ~4.9% to fund it.

## Why these numbers

There is no ground truth to fit here. Cricsheet records what happened, never what
anyone was TRYING to do, so unlike the day-factor or the calibration constants
there is no real-world target to recover -- these are chosen, not discovered.

30% is anchored to a real yardstick so it isn't arbitrary: the league six-rate
moved 4.4% -> 7.6% across 2008-15 vs 2023-26, i.e. fifteen years of the sport
changing. One Attack call moving sixes by 30% is roughly a third of that -- felt,
but not more powerful than the game reinventing itself. Rotate/Contain sit at half
strength: they are a tempo change, not a gamble.

Known asymmetry, deliberately left as-is for now: Attack buys +2.6% boundary for
+0.6% wicket, so it is a favourable bet rather than a true gamble. Raising only
Out's dial would fix that, at the cost of the single-dial simplicity. The harness
is the arbiter -- if all-Attack innings score unrealistically well, that is the
evidence to change it.

## Mirrors

Bowling Attack is numerically IDENTICAL to batting Attack (both make the over more
explosive: more boundaries AND more wickets), and bowling Defend identical to
batting Defend. The mirroring shows up in how they interact, not in the transfers:

    batter Attack  vs  bowler Defend   -> cancel exactly
    batter Defend  vs  bowler Attack   -> cancel exactly
    batter Rotate  vs  bowler Contain  -> cancel exactly
    batter Attack  vs  bowler Attack   -> compounds (carnage or collapse)
    batter Defend  vs  bowler Defend   -> compounds (a dead over)

Cancellation is EXACT only because both sides' deltas are computed off the same
pre-role weights and then added -- never applied one after the other, which would
let the first move change the base the second is measured against.
"""

from __future__ import annotations

ATTACK_DIAL = 0.30
ROTATE_DIAL = 0.15   # half strength: a tempo change, not a gamble

# gain buckets, pay buckets, dial
BAT_ROLES = {
    "attack": (("4", "6", "Out"), ("0", "1"), ATTACK_DIAL),
    "rotate": (("1", "2"), ("4", "6"), ROTATE_DIAL),
    "defend": (("0", "1"), ("4", "6", "Out"), ATTACK_DIAL),
}

BOWL_ROLES = {
    # same transfer as batting attack -- an attacking bowler also makes the over
    # more explosive in both directions
    "attack": (("4", "6", "Out"), ("0", "1"), ATTACK_DIAL),
    # the exact negation of batting rotate: choke the singles, concede boundaries,
    # leave wicket chance untouched
    "contain": (("4", "6"), ("1", "2"), ROTATE_DIAL),
    # same transfer as batting defend
    "defend": (("0", "1"), ("4", "6", "Out"), ATTACK_DIAL),
}


def role_deltas(weights: dict, spec) -> dict:
    """Per-bucket change for one role, measured off `weights`. Sums to zero."""
    d = {k: 0.0 for k in weights}
    if spec is None:
        return d
    gain, pay, dial = spec
    g_tot = sum(weights.get(k, 0.0) for k in gain)
    p_tot = sum(weights.get(k, 0.0) for k in pay)
    if g_tot <= 0.0 or p_tot <= 0.0:
        return d
    transfer = dial * min(g_tot, p_tot)
    for k in gain:
        d[k] += transfer * (weights.get(k, 0.0) / g_tot)
    for k in pay:
        d[k] -= transfer * (weights.get(k, 0.0) / p_tot)
    return d


def apply_roles(weights: dict, bat_role: str | None = None,
                bowl_role: str | None = None) -> dict:
    """Apply both sides' roles to the model's gold probs.

    Both deltas are measured off the SAME `weights` and then added, which is what
    makes a matched pair (e.g. batter Attack vs bowler Defend) cancel to exactly
    zero instead of approximately.
    """
    base = {k: float(v) for k, v in weights.items()}
    d_bat = role_deltas(base, BAT_ROLES.get(bat_role))
    d_bowl = role_deltas(base, BOWL_ROLES.get(bowl_role))
    result = {k: base[k] + d_bat[k] + d_bowl[k] for k in base}

    # Backstop only. Each pay bucket loses at most `dial` of itself per side, so
    # even two compounding roles cap at ~60% -- never negative. This exists so a
    # future dial change can't silently produce an invalid distribution.
    if any(v < 0.0 for v in result.values()):
        result = {k: max(0.0, v) for k, v in result.items()}
    total = sum(result.values())
    if total > 0:
        scale = sum(base.values()) / total
        if abs(scale - 1.0) > 1e-12:
            result = {k: v * scale for k, v in result.items()}
    return result
