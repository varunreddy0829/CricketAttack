"""Pitch character. RETIRED -- no longer applied on the learned-model path.

Kept for reference and for the reasoning below; nothing imports apply_pitch.
It was removed after measuring it against 115,184 real 2014-2022 balls: every
one of its four labels is derived from a number the model already receives as an
input (`dusty`/`green` from the spin-pace edge, `flat`/`slow` from boundary
share), so applying it double-counted. Mean absolute error in predicted
spin/pace runs-per-ball, per 120 balls: model alone 2.93, model + shape only
2.51, model + full layer 3.34. See the note at the removal site in
ml/runtime/adapter.py.

The classic engine's own PITCH_EFFECTS (src/engine/conditions.py) is unaffected
and stays -- it has no venue knowledge, so it duplicates nothing.

--- original rationale, which held only while the level blocks were the problem:

Pitch character, applied after the model.

The model knows each ground's real SCORING RATE -- two numbers measured from
matches actually played there. It knows nothing about the surface's CHARACTER,
because no dataset records whether a track turns square or seams around. That
gap is what this layer fills, and the split matters:

    how MUCH a ground scores  ->  the model, from real data
    what KIND of cricket it is ->  here, hand-authored

The classic engine's PITCH_EFFECTS could not make that split -- it had no venue
knowledge at all, so its multipliers had to carry the scoring level too. Applied
on top of a model that already knows the level, those level effects double-count.
Measured across all 10 grounds against their real 2023-26 averages:

    full PITCH_EFFECTS   12.3 runs mean error   (+9.4 bias)
    style/shape only      6.5 runs mean error   (+6.4 bias)
    no pitch at all       6.9 runs mean error

So the level blocks were actively harmful, and character alone beats nothing.
This module keeps only character.

Two kinds of effect, both designed to leave the SCORING LEVEL alone:

  MATCHUP (dusty, green) -- who the surface favours. Strengthened well past the
      classic engine's 1-2%, which was far too subtle to notice in play. Roughly
      symmetric: what the favoured bowler gains, the other largely gives back.

  SHAPE (flat, slow) -- how runs arrive rather than how many. A road turns
      singles into boundaries; a slow low track does the reverse, with the ball
      staying in play for ones and twos rather than dying as dots. Tuned to be
      near runs-neutral so a ground's real average survives intact.
"""

from __future__ import annotations

# pitch -> {"vs_spin"/"vs_pace"/"all": {outcome: multiplier}}
PITCH_CHARACTER = {
    # Turns square. Spin bites; pace is blunted on the same surface.
    "dusty": {
        "vs_spin": {"Out": 1.10, "0": 1.04, "4": 0.94, "6": 0.94},
        "vs_pace": {"Out": 0.96, "4": 1.02, "6": 1.02},
    },
    # Nip, bounce and movement. The mirror of dusty.
    "green": {
        "vs_pace": {"Out": 1.10, "0": 1.04, "4": 0.94, "6": 0.94},
        "vs_spin": {"Out": 0.96, "4": 1.02, "6": 1.02},
    },
    # A road. Doesn't favour spin or pace -- it favours the BATTER, and the way
    # that shows up is runs arriving in boundaries instead of ones.
    "flat": {
        "all": {"4": 1.10, "6": 1.10, "1": 0.95, "0": 0.97},
    },
    # Low and hard to time. Boundaries are hard to find, but the ball stays in
    # play -- so ones and twos rise rather than dots. Attritional, not hostile.
    "slow": {
        "all": {"4": 0.90, "6": 0.88, "1": 1.07, "2": 1.07},
    },
}


# How much of the flat/slow "all" block to apply. Those two are the only blocks
# that move the scoring LEVEL rather than its character, and the model already
# knows each ground's real level from data -- so they're damped, not removed.
#
# Swept against the real 2023-26 average score of all 10 grounds (mean absolute
# error, lower is better):
#
#     0.00  6.4      1.00  9.7
#     0.25  6.2  <-- 0.75  9.0
#     0.50  7.6
#
# 0.25 is a genuine optimum: a light touch beats no touch, and anything stronger
# fights the per-ground rates the model already has. Note the flat/slow labels
# themselves disagree with reality on some grounds (Arun Jaitley and Rajiv Gandhi
# are labelled "slow" but are two of the highest-scoring venues), which is why a
# strong level effect actively hurts -- it pushes those grounds the wrong way.
SHAPE_STRENGTH = 0.25


def _damp(mult: float, strength: float) -> float:
    """Pull a multiplier toward 1.0 by `strength` (1.0 = full, 0.0 = no effect)."""
    return 1.0 + (mult - 1.0) * strength


def apply_pitch(weights: dict, pitch: str | None, bowler_style: str | None) -> dict:
    """Apply pitch character. Conserves the incoming weight total.

    Unknown or missing pitch is a no-op, so a ground with no configured surface
    simply gets the model's own view of it.
    """
    spec = PITCH_CHARACTER.get(pitch or "")
    if not spec:
        return weights

    w = {k: float(v) for k, v in weights.items()}
    total_before = sum(w.values())

    style_key = "vs_spin" if bowler_style == "Spin" else "vs_pace"
    # matchup blocks apply at full strength -- they're what the data CAN'T supply.
    # the "all" block is the level-moving one, so it's damped against the model's
    # own per-ground knowledge.
    for block, strength in ((spec.get(style_key), 1.0),
                            (spec.get("all"), SHAPE_STRENGTH)):
        if not block:
            continue
        for k, mult in block.items():
            if k in w:
                w[k] *= _damp(mult, strength)

    # renormalise: these are multipliers, not transfers, so the total drifts.
    # Rescaling keeps it a valid distribution and means a pitch shifts the SHAPE
    # of the odds without quietly adding or removing probability mass.
    total_after = sum(w.values())
    if total_after > 0:
        f = total_before / total_after
        return {k: v * f for k, v in w.items()}
    return weights
