"""Turn a base-weights provider into a drop-in `calculate_single_ball`.

The engine's pipeline is: base -> player stages -> conditions -> roles -> cascade
-> sample. This adapter replaces everything up to and including the pitch/phase
half of conditions, and leaves every player-facing lever exactly as the classic
engine applies it.

    base_provider(striker, bowler, ctx) -> 8-key dict summing to 1000.0

The Phase 1 lookup table and the Phase 2 learned model are both just providers, so
the harness, the shadow server and the tests never change between them.

Nothing here modifies src/. `apply_conditions` is called with `pitch=None` and
`over_num=None`: reading src/engine/conditions.py, `PITCH_EFFECTS.get(None)` is
falsy and the phase block is guarded by `is not None`, while both gambit branches
still fire. That skips exactly the two world-state effects the base already
accounts for, and keeps the one-shot cards intact -- with no source edit and no
file split.
"""

from __future__ import annotations

import random

from src.engine.conditions import apply_conditions
from src.engine.roles import apply_modes, apply_roles
from ml.runtime.roles import apply_roles as ml_apply_roles
from src.engine.simulator import WICKET_CASCADE_MULT
from src.engine.stats_calculator import (
    apply_stage1_ovr,
    apply_stage2_strike_rate_economy,
    apply_stage3_wicket_factor,
)

ENGINE_KEYS = ("0", "1", "2", "3", "4", "5", "6", "Out")


def _post_model_conditions(ctx: dict | None) -> dict | None:
    """What still applies AFTER the model: pitch character and the gambit cards.

    Two of the three things `apply_conditions` handles survive the model, one
    doesn't:

      pitch      KEPT. The model knows each ground's real SCORING RATE (two
                 numbers measured from actual matches there), but nothing about
                 its CHARACTER -- no dataset records whether a surface turns
                 square or seams, so it cannot know a dusty track helps spinners.
                 That is hand-authored knowledge the data can't supply, which is
                 exactly the kind that belongs in a post-model layer.
      gambits    KEPT. One-shot player choices; never inferable from history.
      over_num   DROPPED. This drives PHASE_EFFECTS (powerplay/death), and the
                 model already takes the over as 20 separate inputs -- it learned
                 each over's real shape, including that the old hand-tuned
                 powerplay multipliers had two SIGNS INVERTED (they raised wicket
                 chance and cut dots; reality is the opposite on both).

    Returns None when there's nothing to apply, so the common case skips the call.
    """
    if not ctx:
        return None
    pitch = ctx.get("pitch")
    if not pitch and not (ctx.get("attack_gambit") or ctx.get("trap_gambit")):
        return None
    return {
        "pitch": pitch,
        "bowler_style": ctx.get("bowler_style"),
        "over_num": None,
        "attack_gambit": ctx.get("attack_gambit"),
        "trap_gambit": ctx.get("trap_gambit"),
        "striker_intent": ctx.get("striker_intent", 50),
    }


def normalise(weights: dict, total: float = 1000.0) -> dict:
    s = sum(weights.values())
    if s <= 0:
        return {k: total / len(weights) for k in weights}
    return {k: v / s * total for k, v in weights.items()}


def make_ball_fn(
    base_provider,
    *,
    player_stages: bool = False,
    cascade: bool = True,
    calibration: float = 1.0,
    out_calibration: float = 1.0,
    new_roles: bool = True,
):
    """Build a `calculate_single_ball`-shaped function.

    player_stages: run the classic Stage 1/2/3 player-quality ratios on top of the
        base. True for the Phase 1 lookup (whose base is a league average and needs
        per-player differentiation); False for the learned model, which already
        knows who is batting.
    cascade: apply the engine's Stage 6 wicket cascade. Kept on by default; Phase 4
        measures whether a wickets-aware base subsumes it.
    calibration: global multiplier on the scoring buckets.
    out_calibration: global multiplier on Out.

        Together these absorb the inflation the downstream bonuses add on top of a
        situation-aware base. They matter far more than they look: the wicket rate
        compounds over 120 balls, so a 1.17x per-ball Out error becomes a 5x error
        in the all-out rate. Fitted by ml/harness/calibrate.py, not guessed.
    """

    scale = {k: calibration for k in ("1", "2", "3", "4", "5", "6")}
    scale["Out"] = out_calibration
    needs_scaling = calibration != 1.0 or out_calibration != 1.0

    def ball_fn(striker, bowler, league_avg, context=None):
        ctx = context or {}
        w = base_provider(striker, bowler, ctx)

        if player_stages:
            w = apply_stage1_ovr(w, striker, bowler)
            w = apply_stage2_strike_rate_economy(w, striker, bowler, league_avg)
            w = apply_stage3_wicket_factor(w, striker, bowler, league_avg)

        if needs_scaling:
            w = {k: v * scale.get(k, 1.0) for k, v in w.items()}
            w = normalise(w)

        g = _post_model_conditions(ctx)
        if g:
            w = apply_conditions(w, g)

        if ctx.get("bat_role") or ctx.get("bowl_role"):
            if new_roles:
                # ml/runtime/roles.py -- explicit paired transfers. Replaces the
                # classic engine's multiply-and-let-dots-absorb approach, which
                # drifted Out on Rotate/Contain (meant to be risk-neutral) and let
                # 1s/2s rise alongside boundaries on Attack instead of paying for
                # them. No skill-grid bonus here either: the model already takes
                # all 9 grid cells as input, so re-applying it would count the
                # same skill twice.
                w = ml_apply_roles(w, bat_role=ctx.get("bat_role"),
                                   bowl_role=ctx.get("bowl_role"))
            else:
                w = apply_modes(w, bat_mode=ctx.get("bat_role"),
                                bowl_mode=ctx.get("bowl_role"))
                w = apply_roles(
                    w,
                    bat_role=ctx.get("bat_role"), bat_grid=ctx.get("bat_grid", 50),
                    bowl_role=ctx.get("bowl_role"), bowl_grid=ctx.get("bowl_grid", 50),
                )

        if cascade:
            n = ctx.get("wickets_this_over", 0)
            if n > 0:
                removed = w["Out"] * (1.0 - WICKET_CASCADE_MULT ** n)
                w = dict(w)
                w["Out"] -= removed
                w["0"] += removed

        keys = list(w.keys())
        vals = [w[k] for k in keys]
        total = sum(vals)
        probs = [v / total for v in vals] if total > 0 else [1.0 / len(vals)] * len(vals)
        return random.choices(keys, weights=probs, k=1)[0]

    return ball_fn
