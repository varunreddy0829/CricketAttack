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
        "vs_spin": {"Out": 1.12, "0": 1.08},
        "vs_pace": {"Out": 0.96},
        "all": {"4": 0.92, "6": 0.92},
    },
    "green": {
        "vs_pace": {"Out": 1.12, "0": 1.08},
        "vs_spin": {"Out": 0.96},
        "all": {"4": 0.95},
    },
    "flat": {
        "all": {"4": 1.15, "6": 1.15, "Out": 0.88, "0": 0.92},
    },
    "slow": {
        "vs_spin": {"Out": 1.08},
        # 4/6 cut hard; 1/2 bumped up more than 0 -- still genuinely
        # low-scoring (no boundaries), but the ball stays in play for a
        # single/double more often than it just dies as a dot.
        "all": {"4": 0.88, "6": 0.85, "0": 1.06, "1": 1.12, "2": 1.12},
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

# Gambits: one-shot per-match cards, armed secretly with the over submission,
# in effect for exactly the one over they're used on. Each applies two
# EXACT, independent, percentage adjustments -- a boost to its side's stat
# and a cut to the opposing side's -- then settles whatever's left over
# against '0', the one flexible bucket every gambit draws from or returns to.
# 1/2/3/5 (and, for Attack, nothing else on the bowling side; for Trap,
# nothing else on the batting side) are never touched. See _apply_gambit_*.
GAMBIT_ATTACK_BOOST = 1.5     # '4' and '6' each get 50% more weight
GAMBIT_ATTACK_OUT_CUT = 0.25  # Out drops by 25% on top
# Trap Set reads the striker's effective intent for the ball: punishes
# aggression hard, mildly pressures neutral play, and is wasted on blockers
# (intent<=40 actually GIVES weight back to '0' -- a trap sprung on a
# blocker is a wasted card, not a bonus). The boundary cut is flat regardless
# of intent -- it's the bowler's field/plan containing the shot, not a read
# on the batter.
TRAP_VS_AGGRESSIVE = 1.6   # striker intent >= 60
TRAP_VS_NEUTRAL = 1.2
TRAP_VS_DEFENSIVE = 0.75   # striker intent <= 40
GAMBIT_TRAP_BOUNDARY_CUT = 0.25   # '4' and '6' each drop by 25% on top


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


def _settle_with_zero(w, net_change):
    """Whatever a gambit's direct boost/cut doesn't already balance out gets
    settled against '0' -- taken from it if the net change needs funding,
    or returned to it if the cut alone overshoots. Keeps the total exactly
    conserved without touching any key beyond the ones already adjusted."""
    if net_change >= 0:
        take = min(w.get("0", 0), net_change)
        w["0"] = w.get("0", 0) - take
    else:
        w["0"] = w.get("0", 0) + (-net_change)


def _apply_gambit_attack(w):
    add4 = w["4"] * (GAMBIT_ATTACK_BOOST - 1)
    add6 = w["6"] * (GAMBIT_ATTACK_BOOST - 1)
    cut_out = w["Out"] * GAMBIT_ATTACK_OUT_CUT
    w["4"] += add4
    w["6"] += add6
    w["Out"] -= cut_out
    _settle_with_zero(w, (add4 + add6) - cut_out)


def _apply_gambit_trap(w, intent):
    if intent >= 60:
        mult = TRAP_VS_AGGRESSIVE
    elif intent <= 40:
        mult = TRAP_VS_DEFENSIVE
    else:
        mult = TRAP_VS_NEUTRAL
    add_out = w["Out"] * (mult - 1)   # negative when mult < 1 (wasted on a blocker)
    cut4 = w["4"] * GAMBIT_TRAP_BOUNDARY_CUT
    cut6 = w["6"] * GAMBIT_TRAP_BOUNDARY_CUT
    w["Out"] += add_out
    w["4"] -= cut4
    w["6"] -= cut6
    _settle_with_zero(w, add_out - (cut4 + cut6))


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
        _apply_gambit_attack(w)

    if ctx.get("trap_gambit"):
        _apply_gambit_trap(w, ctx.get("striker_intent", 50))

    total = sum(w.values())
    if total <= 0:
        return dict(weights)
    factor = original_sum / total
    return {k: v * factor for k, v in w.items()}
