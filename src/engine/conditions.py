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
# Each is a DIRECT transfer of weight out of '0' into the buffed stat(s) --
# not a plain multiply left for the final renormalization to sort out, which
# would have quietly shaved a bit off every other category too (including
# the ones a gambit is explicitly supposed to leave alone, like Out on
# Attack or 4/5/6 on Trap). See _apply_gambit_* below.
GAMBIT_ATTACK_BOOST = 1.5   # '4' and '6' each get 50% more weight, paid for by '0'
# Trap Set reads the striker's effective intent for the ball: punishes
# aggression hard, mildly pressures neutral play, and is wasted on blockers
# (intent<=40 actually GIVES weight back to '0' -- a trap sprung on a
# blocker is a wasted card, not a bonus).
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


def _transfer(w, src, dst_delta):
    """Move exactly sum(dst_delta.values()) of weight from w[src] into the
    dst keys (each grown by its own delta) -- a true zero-sum swap between
    two specific buckets, immune to whatever the final global renormalize
    does to everything else. Caps at what `src` actually has (only matters
    in extreme edge cases where an earlier stage already crushed it)."""
    need = sum(dst_delta.values())
    if need <= 0:
        w[src] = w.get(src, 0) - need   # negative need = giving weight back to src
        for k, d in dst_delta.items():
            w[k] = w.get(k, 0) + d
        return
    take = min(w.get(src, 0), need)
    scale = take / need if need else 0
    for k, d in dst_delta.items():
        w[k] = w.get(k, 0) + d * scale
    w[src] = w.get(src, 0) - take


def _apply_gambit_attack(w):
    add4 = w["4"] * (GAMBIT_ATTACK_BOOST - 1)
    add6 = w["6"] * (GAMBIT_ATTACK_BOOST - 1)
    _transfer(w, "0", {"4": add4, "6": add6})


def _apply_gambit_trap(w, intent):
    if intent >= 60:
        mult = TRAP_VS_AGGRESSIVE
    elif intent <= 40:
        mult = TRAP_VS_DEFENSIVE
    else:
        mult = TRAP_VS_NEUTRAL
    add_out = w["Out"] * (mult - 1)   # negative when mult < 1 (wasted on a blocker)
    _transfer(w, "0", {"Out": add_out})


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
