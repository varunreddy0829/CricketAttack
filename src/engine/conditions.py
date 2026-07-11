# src/engine/conditions.py
#
# Stage 3.5 — match conditions. Applied between the wicket factor (stage 3)
# and intents (stage 4, always last): pitch personality, innings phase,
# dot-ball pressure, and one-shot gambit cards. Like every other stage it
# conserves the incoming weight sum (1000.0) and leaves the rare boundary
# types '3' and '5' untouched (they're near-static constants by design).

# Per-pitch multipliers. "vs_spin"/"vs_pace" apply only when the bowler's
# style matches; "all" applies regardless of who's bowling.
PITCH_EFFECTS = {
    "dusty": {
        "vs_spin": {"Out": 1.18, "0": 1.08},
        "vs_pace": {"Out": 0.92},
        "all": {"4": 0.92, "6": 0.92},
    },
    "green": {
        "vs_pace": {"Out": 1.18, "0": 1.08},
        "vs_spin": {"Out": 0.92},
        "all": {"4": 0.95},
    },
    "flat": {
        "all": {"4": 1.15, "6": 1.15, "Out": 0.88, "0": 0.92},
    },
    "slow": {
        "vs_spin": {"Out": 1.08},
        "all": {"4": 0.88, "6": 0.85, "0": 1.12, "1": 1.05},
    },
}

# Innings phases (over_num is 0-indexed). Powerplay: fielders up, boundaries
# flow but mishits carry to the ring. Death: everything swings for the fence.
PHASE_POWERPLAY_END = 6    # overs 1-6
PHASE_DEATH_START = 15     # overs 16-20
PHASE_EFFECTS = {
    "powerplay": {"4": 1.25, "6": 1.10, "Out": 1.10, "0": 0.95},
    "middle": {},
    "death": {"4": 1.15, "6": 1.30, "Out": 1.25, "0": 0.90, "1": 0.92},
}

# Pressure: consecutive dot balls squeeze the batter — each stacked dot adds
# to the wicket chance until something releases it (see server-side tracking).
PRESSURE_OUT_PER_DOT = 0.05
PRESSURE_CAP = 6

# Gambits: one-shot per-match cards, armed secretly with the over submission.
# Each buffs exactly one side's stat and nothing else -- no paired nerf on
# top (any tiny redistribution elsewhere is just the renormalization every
# stage already does to keep weights summing to 1000, not a deliberate hit).
GAMBIT_ATTACK = {"4": 1.5, "6": 1.5}
# Trap Set reads the striker's effective intent for the ball: punishes
# aggression hard, mildly pressures neutral play, and is wasted on blockers.
TRAP_VS_AGGRESSIVE = 1.6   # striker intent >= 60
TRAP_VS_NEUTRAL = 1.2
TRAP_VS_DEFENSIVE = 0.75   # striker intent <= 40


def phase_for_over(over_num):
    if over_num < PHASE_POWERPLAY_END:
        return "powerplay"
    if over_num >= PHASE_DEATH_START:
        return "death"
    return "middle"


def _apply_mults(weights, mults):
    for k, m in mults.items():
        if k in weights:
            weights[k] *= m


def apply_conditions(weights, ctx):
    """ctx keys (all optional): pitch, bowler_style, over_num, pressure,
    attack_gambit (bool), trap_gambit (bool), striker_intent."""
    if not ctx:
        return dict(weights)
    original_sum = sum(weights.values())
    w = dict(weights)

    pitch = PITCH_EFFECTS.get(ctx.get("pitch"))
    if pitch:
        style_key = "vs_spin" if ctx.get("bowler_style") == "Spin" else "vs_pace"
        _apply_mults(w, pitch.get(style_key, {}))
        _apply_mults(w, pitch.get("all", {}))

    if ctx.get("over_num") is not None:
        _apply_mults(w, PHASE_EFFECTS[phase_for_over(ctx["over_num"])])

    pressure = min(int(ctx.get("pressure") or 0), PRESSURE_CAP)
    if pressure > 0:
        w["Out"] *= 1.0 + PRESSURE_OUT_PER_DOT * pressure

    if ctx.get("attack_gambit"):
        _apply_mults(w, GAMBIT_ATTACK)

    if ctx.get("trap_gambit"):
        intent = ctx.get("striker_intent", 50)
        if intent >= 60:
            w["Out"] *= TRAP_VS_AGGRESSIVE
        elif intent <= 40:
            w["Out"] *= TRAP_VS_DEFENSIVE
        else:
            w["Out"] *= TRAP_VS_NEUTRAL

    total = sum(w.values())
    if total <= 0:
        return dict(weights)
    factor = original_sum / total
    return {k: v * factor for k, v in w.items()}
