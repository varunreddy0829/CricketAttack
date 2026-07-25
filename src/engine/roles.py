# src/engine/roles.py
"""
Stage 4 (roles) — the batting/bowling role step that replaces the old intent
slider. It runs LAST in the pipeline (after conditions, before sampling) and
bends the 1000-unit weight bucket with two targeted, grid-scaled transfers:
the batter's chosen role and the bowler's chosen role.

Design (locked):

  advantage(grid, ceil) = ceil * clamp((grid - 50) / 49, 0, 1) ** 2
    - grid  = the player's 0-99 Playstyle-Grid value for the current (role,
              phase) cell. 50 = league median (no bonus), 99 = best (full).
    - Squared => flat through the middle, steep only at the elite tail.
    - `ceil` is the per-role dial (differs by role, see *_CEIL below).

Batting roles (transfer measured off the pre-role weights):
  - attack : 4s & 6s  x (1 + adv)  [cap x1.5], paid from dots. Out untouched.
  - rotate : 1s & 2s  x (1 + adv)  [cap x1.25], paid from dots.
  - defend : Out       x (1 - adv)  [cap x0.5], freed weight split 50/50 dots & 1s.

Bowling roles are the EXACT INVERSE transfer of the batting role they counter:
  - attack (counters batting defend) : Out x (1 + adv), pulled 50/50 from dots & 1s.
  - contain(counters batting rotate) : 1s & 2s x (1 - adv), pushed to dots.
  - defend (counters batting attack) : 4s & 6s x (1 - adv), pushed to dots.

Because both sides are measured off the SAME base and each role's transfers sum
to zero, a matched counter cancels by the grid difference (equal specialists ->
baseline; better grid keeps the surplus) and a mismatch lets both land. The
1000 sum is conserved by construction; a rare extreme ball that would drive a
bucket negative is clamped to 0 and renormalised (the "dot floor" backstop).
"""

# Per-role ceilings for the advantage formula. Attack/Defend reach 0.5 (a x1.5
# boundary boost / a halved Out); Rotate is gentler at 0.25 (singles are the
# biggest base, so x1.25 is already a large absolute move — see design note).
BAT_CEIL = {"attack": 0.5, "rotate": 0.25, "defend": 0.5}
# Bowling ceilings mirror the batting role each one counters, so equal grids
# cancel exactly: attack<->defend at 0.5, contain<->rotate at 0.25.
BOWL_CEIL = {"attack": 0.5, "contain": 0.25, "defend": 0.5}

BAT_ROLES = frozenset(BAT_CEIL)
BOWL_ROLES = frozenset(BOWL_CEIL)


def role_advantage(grid, ceil):
    """advantage = ceil * clamp((grid-50)/49, 0, 1)**2 — zero at/below the median."""
    if grid is None or grid <= 50.0 or ceil <= 0.0:
        return 0.0
    x = (float(grid) - 50.0) / 49.0
    if x > 1.0:
        x = 1.0
    return ceil * x * x


def _blank_deltas(base):
    return {k: 0.0 for k in base}


def batting_role_deltas(base, role, grid):
    """Per-bucket weight change for a batting role, measured off `base`.
    Returns a dict of deltas that sums to 0 (an internal transfer)."""
    d = _blank_deltas(base)
    if role not in BAT_ROLES:
        return d
    adv = role_advantage(grid, BAT_CEIL[role])
    if adv <= 0.0:
        return d
    if role == "attack":                       # boundaries up, from dots
        gain = base["4"] * adv + base["6"] * adv
        d["4"] += base["4"] * adv
        d["6"] += base["6"] * adv
        d["0"] -= gain
    elif role == "rotate":                     # singles up, from dots
        gain = base["1"] * adv + base["2"] * adv
        d["1"] += base["1"] * adv
        d["2"] += base["2"] * adv
        d["0"] -= gain
    elif role == "defend":                     # Out down, freed split 50/50 dots & 1s
        freed = base["Out"] * adv
        d["Out"] -= freed
        d["0"] += freed / 2.0
        d["1"] += freed / 2.0
    return d


def bowling_role_deltas(base, role, grid):
    """Per-bucket weight change for a bowling role, measured off `base`.
    Each is the exact inverse of the batting role it counters; sums to 0."""
    d = _blank_deltas(base)
    if role not in BOWL_ROLES:
        return d
    adv = role_advantage(grid, BOWL_CEIL[role])
    if adv <= 0.0:
        return d
    if role == "attack":                       # Out up, pulled 50/50 from dots & 1s
        add = base["Out"] * adv
        d["Out"] += add
        d["0"] -= add / 2.0
        d["1"] -= add / 2.0
    elif role == "contain":                    # singles down, into dots
        cut = base["1"] * adv + base["2"] * adv
        d["1"] -= base["1"] * adv
        d["2"] -= base["2"] * adv
        d["0"] += cut
    elif role == "defend":                     # boundaries down, into dots
        cut = base["4"] * adv + base["6"] * adv
        d["4"] -= base["4"] * adv
        d["6"] -= base["6"] * adv
        d["0"] += cut
    return d


def _finalise(result):
    """Guarantee a valid 1000-sum distribution. In the normal case every bucket
    is already >= 0 and the sum is exactly conserved (deltas balance), so this
    is a no-op. The clamp+renormalise only fires on a rare extreme ball where a
    withdrawal would drive a bucket below 0 — the 'dot floor' backstop."""
    if any(v < 0.0 for v in result.values()):
        for k in result:
            if result[k] < 0.0:
                result[k] = 0.0
        total = sum(result.values())
        if total > 0.0:
            return {k: v / total * 1000.0 for k, v in result.items()}
    return result


def apply_roles(weights, bat_role=None, bat_grid=50, bowl_role=None, bowl_grid=50):
    """Apply both roles off the same pre-role `weights` and return the new
    1000-unit distribution. Either role may be None (that side stays neutral)."""
    base = {k: float(v) for k, v in weights.items()}
    bd = batting_role_deltas(base, bat_role, bat_grid) if bat_role else _blank_deltas(base)
    wd = bowling_role_deltas(base, bowl_role, bowl_grid) if bowl_role else _blank_deltas(base)
    result = {k: base[k] + bd[k] + wd[k] for k in base}
    return _finalise(result)


def apply_batting_role(weights, role, grid):
    """Batting role only (bowler neutral) — mostly for tests / standalone use."""
    return apply_roles(weights, bat_role=role, bat_grid=grid)


def apply_bowling_role(weights, role, grid):
    """Bowling role only (batter neutral) — mostly for tests / standalone use."""
    return apply_roles(weights, bowl_role=role, bowl_grid=grid)
