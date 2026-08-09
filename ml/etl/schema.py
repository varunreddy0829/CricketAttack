"""Feature layout. The single source of truth for column order and player anchors.

`ml/runtime/features.py` is the only place that writes a feature row, and it
imports its layout from here. Both the offline ETL and the live adapter call it,
so train/serve skew -- the standard way projects like this fail silently -- is
structurally impossible rather than merely avoided by discipline.

A SCHEMA_HASH travels with every trained model. Loading a model whose hash doesn't
match this file fails loudly instead of silently mis-ordering columns.
"""

from __future__ import annotations

import hashlib

# --- context features ------------------------------------------------------
# Over is one-hot rather than linear: the powerplay-middle-death shape is sharply
# nonlinear and 20 free coefficients recover whatever the data actually says,
# without needing a hidden layer. Same idea for the required-rate thresholds,
# which turn a linear model into a piecewise-linear one.

N_OVERS = 20

# Balls faced by the striker, as one-hot buckets rather than a capped ramp.
# Measured over 124,765 legal balls in 2014-2022, runs/ball by balls faced is
# 0.806 -> 1.243 (5) -> 1.396 (6-14) -> 1.440 (15-29) -> 1.620 (30-59): steep
# early, flat through the middle, rising again. That shape is not linear and the
# old min(balls,15)/15 cap threw away everything past ball 15. Buckets let the
# data pick the curve, exactly as the 20 over one-hots do.
#
# The old cap was justified to me as guarding against survivorship making set
# batters look invulnerable. The data does not support that: out% is FLAT to
# RISING with balls faced (4.81% at 15-29, 5.94% at 30-59), so dismissal keeps
# pulling batters out and the feedback loop under simulation stays bounded.
STRIKER_BALL_BUCKETS = [
    (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
    (6, 9), (10, 14), (15, 24), (25, 39), (40, 10_000),
]

CONTEXT_FEATURES = [
    *[f"over_{i:02d}" for i in range(N_OVERS)],
    "ball_in_over",         # /6
    "wickets",              # /10
    "is_second_innings",
    "rrr",                  # /15, clipped; 0 in the first innings
    "rrr_gt_8",
    "rrr_gt_12",
    "rrr_gt_16",
    "wickets_x_death",      # wickets down matters far more at the death
    *[f"sb_{lo}_{hi}" for lo, hi in STRIKER_BALL_BUCKETS],
    "striker_position",     # /11
    "bowler_balls",         # /24
    "over_in_spell",        # /4
    "bat_career_balls",     # log1p/10 -- tells the model when to hedge
    "bowl_career_balls",    # log1p/10
    "nonstriker_sr",        # /200

    # --- interactions: computed per ball, where BOTH sides are known ---------
    # These exist because the model is additive -- logits = a + row.B
    # + E_bat.V_bat + E_bowl.V_bowl -- and E_bat is a lookup by player index
    # alone. A number sitting in a player's anchor is therefore IDENTICAL in
    # over 1 and over 19, and identical against spin and pace. Any "this player
    # is good at X" fact must be resolved against the live situation here, in
    # the context row, or it is structurally inert.
    "bat_sr_edge_vs_type",     # vs THIS bowler's type, minus his own overall
    "bat_out_edge_vs_type",
    "bat_bdry_edge_vs_type",   # separates a spin BASHER from a spin milker:
                               # Gambhir's SR vs spin is +22 while his boundary
                               # rate FALLS -- he runs it, he doesn't hit it
    "bat_phase_sr_edge",       # in THIS phase, minus his own overall
    "bowl_phase_eco_edge",

    # --- venue: level, shape, and who it favours -----------------------------
    "venue_runs_per_ball",  # /2,  leave-one-match-out
    "venue_wkts_per_ball",  # x20, leave-one-match-out
    "venue_bdry_share",     # boundary runs / all runs -- the road<->grind axis,
                            # independent of level: Jaipur and Wankhede score
                            # alike, but Jaipur runs them and Wankhede hits them
    "venue_type_edge",      # this ground's economy edge for THIS bowler's type,
                            # residualised (each bowler vs his own economy) so it
                            # measures the pitch and not who happened to bowl on it
]

N_CONTEXT = len(CONTEXT_FEATURES)
CTX = {name: i for i, name in enumerate(CONTEXT_FEATURES)}

# Features deliberately NOT here, and why:
#
#   striker_current_runs   A direct readout of "this batter is having a good day".
#                          Under simulation it is driven by the model's own samples,
#                          so a hot start begets a hotter one -- a hot-hand feedback
#                          loop stacked on top of survivorship bias.
#
#   balls_since_boundary,  Not leaky (both are computable at inference), but
#   wickets_in_last_12     ENDOGENOUS under simulation. In training a drought
#                          proxies for hard conditions; in simulation the drought is
#                          caused by the model's own draws, so a random quiet patch
#                          lowers boundary probability, which lengthens the quiet
#                          patch. Positive feedback with the wrong sign: it
#                          manufactures collapses. The second is also already
#                          modelled by the engine's Stage 6 cascade.
#
#   score / current RR     Same endogeneity, first-innings flavour. The chase side
#                          keeps RRR, where the feedback (fall behind -> take risks
#                          -> lose wickets) is real cricket and the target itself is
#                          exogenous. Over, wickets and batter identity carry nearly
#                          all the first-innings information anyway.
#
#   raw striker_balls      Capped at 15 with a binary `is_set` instead. A batter on
#                          ball 31 has BY CONSTRUCTION survived 30, so most of the
#                          raw slope is survivorship, not "getting your eye in".
#                          Uncapped, tailenders start making hundreds.
#
#   batter x bowler        811x811 pairs over 283k balls is ~0.4 balls per cell.
#   interaction            Anything fitted there is noise, and it invents matchups
#                          that never happened. Additivity is the feature.

# --- player anchors --------------------------------------------------------
# Players enter as e_p = W . f_p + delta_p: a projection of these observables plus
# a small learned correction. That is what makes cold start work -- a player who
# was never in training (the auction pool allows anyone) still gets a sensible
# vector from f_p alone, and the long tail of sub-100-ball players shrinks toward
# it rather than fitting noise.

# `ovr` is NOT here, and its absence is load-bearing. Era OVRs are DERIVED from
# this model by ml/train/derive_ovr.py, so feeding them back in is circular. It
# was previously pinned to the constant 55 for every era player, which made it a
# dead column that contributed nothing beyond the intercept -- while the matching
# `nonstriker_ovr` context feature was pinned at train time but read the REAL
# rating at serve time, a train/serve skew worth ~4% on runs per ball. Deleting
# the feature removes the bug by construction rather than guarding against it.
BAT_ANCHOR = [
    "log_balls", "sr", "avg", "four_rate", "six_rate", "out_rate",
    "sf_pp_attack", "sf_pp_anchor", "sf_pp_rotate",
    "sf_mid_attack", "sf_mid_anchor", "sf_mid_rotate",
    "sf_death_attack", "sf_death_anchor", "sf_death_rotate",
]

# `sr` (balls per wicket) is gone too: it is the EXACT reciprocal of wkt_rate
# (wickets per ball), so the two carried one fact between them and merely split
# a coefficient inside a rank-4 bottleneck.
BOWL_ANCHOR = [
    "log_balls", "eco", "avg", "wkt_rate", "is_spin",
    "bf_pp_attack", "bf_pp_contain", "bf_pp_defend",
    "bf_mid_attack", "bf_mid_contain", "bf_mid_defend",
    "bf_death_attack", "bf_death_contain", "bf_death_defend",
]

N_BAT_ANCHOR = len(BAT_ANCHOR)
N_BOWL_ANCHOR = len(BOWL_ANCHOR)

ENTITY_RANK = 4      # low-rank player effect; 16 free dims per player fits noise

# --- outcome classes (mirrors ml.etl.replay.CLASSES) -----------------------
CLASSES = ("0", "1", "2", "3", "4", "6", "Out", "wide", "noball")
N_CLASSES = len(CLASSES)

# The 8-key dict the engine pipeline expects, given a legal delivery. '5' is not
# modelled; it is injected as a constant so the contract stays intact.
ENGINE_KEYS = ("0", "1", "2", "3", "4", "5", "6", "Out")
CONSTANT_FIVE_WEIGHT = 1.0     # out of 1000, matching src/engine BASELINE_WEIGHTS


def schema_hash() -> str:
    payload = "|".join([
        *CONTEXT_FEATURES, "#", *BAT_ANCHOR, "#", *BOWL_ANCHOR, "#", *CLASSES,
        f"#rank={ENTITY_RANK}",
    ])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


SCHEMA_HASH = schema_hash()

if __name__ == "__main__":
    print(f"context features : {N_CONTEXT}")
    print(f"bat anchor       : {N_BAT_ANCHOR}")
    print(f"bowl anchor      : {N_BOWL_ANCHOR}")
    print(f"classes          : {N_CLASSES}  {CLASSES}")
    print(f"schema hash      : {SCHEMA_HASH}")
