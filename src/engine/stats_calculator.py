# src/engine/stats_calculator.py

from src.models.player import Batter, Bowler

# Stage 0: Initial Baseline Weights (sum to 1000). This is the single starting
# point for EVERY batter -- no per-batter DNA step. A prior per-batter "DNA"
# stage (deriving a distribution from the batter's own dismissals/fours/sixes
# rate before the bowler was even considered) was removed: it let a batter's
# own career averages double-count against what Stage 3's patience/defense
# already computes from the same underlying stats, distorting Out probabilities
# both ways (e.g. inflating an elite batter's safety and a weak batter's
# fragility well beyond what the comparative bowler-vs-batter stages warrant).
BASELINE_WEIGHTS = {
    '0': 337.0,
    '1': 354.0,
    '2': 65.0,
    '3': 4.0,
    '4': 121.0,
    '5': 1.0,
    '6': 63.0,
    'Out': 55.0
}

def apply_stage1_ovr(weights: dict, batter: Batter, bowler: Bowler) -> dict:
    """
    Stage 1: OVR Strength Factor
    Compare batsman total OVR / bowler total OVR.
    If > 1: increase 1, 2, 4, 6 (multiply by OVR ratio), decrease 0, Out (adjust proportionally to conserve 1000 sum).
    If < 1: decrease 1, 2, 4, 6 (multiply by OVR ratio), increase 0, Out (adjust proportionally to conserve 1000 sum).
    """
    r_ovr = batter.ovr / bowler.ovr if bowler.ovr > 0 else 1.0
    
    # Store old values to maintain total sum
    result = {k: float(v) for k, v in weights.items()}
    
    # Modify 1, 2, 4, 6
    for k in ['1', '2', '4', '6']:
        result[k] *= r_ovr

    # Elements remaining constant (3 and 5 are static rare events)
    static_sum = result['3'] + result['5']
    target_active_sum = 1000.0 - static_sum
    
    active_modified_sum = result['1'] + result['2'] + result['4'] + result['6']
    target_0_out_sum = target_active_sum - active_modified_sum

    if target_0_out_sum >= 0:
        # Distribute target_0_out_sum to '0' and 'Out' proportionally
        old_0_out_sum = weights['0'] + weights['Out']
        if old_0_out_sum > 0:
            result['0'] = target_0_out_sum * (weights['0'] / old_0_out_sum)
            result['Out'] = target_0_out_sum * (weights['Out'] / old_0_out_sum)
        else:
            # If they were both 0, split evenly
            result['0'] = target_0_out_sum * 0.9
            result['Out'] = target_0_out_sum * 0.1
    else:
        # The modified (1, 2, 4, 6) sum exceeds the budget
        result['0'] = 0.0
        result['Out'] = 0.0
        # Scale down (1, 2, 4, 6) to fit exactly into target_active_sum
        scale = target_active_sum / active_modified_sum
        for k in ['1', '2', '4', '6']:
            result[k] *= scale

    # Normalize to exactly 1000.0 units
    total = sum(result.values())
    if total > 0.0:
        result = {k: (v / total) * 1000.0 for k, v in result.items()}
    return result

def apply_stage2_strike_rate_economy(weights: dict, batter: Batter, bowler: Bowler, league_avg: dict) -> dict:
    """
    Stage 2: Strike Rate vs Economy Factor (P_str) -- longevity-aware + ghost-stat smoothed.

    Same one-season-wonder problem as Stage 3 had: raw strike_rate/economy have
    no volume weighting, so a batter with a handful of freakish balls reads as a
    better striker than a proven veteran with a sustainable rate. Both sides use
    a VOLUME^A x RATE^B "power" score, ghost-stat smoothed toward a league prior:
        ghost_bat_sr = (runs + K_bs*prior_bat_sr/100) / (balls + K_bs) * 100
        ghost_eco    = (runs_conceded + K_be*prior_eco) / (legal_balls + K_be) * 6
        bat_power  = ( balls^A_bat        * ghost_bat_sr^B_bat  ) / bat_power_base
        bowl_power = ( legal_balls^A_bowl / ghost_eco^B_bowl    ) / bowl_power_base
        P_str = bat_power / bowl_power
    If > 1: increase 1, 2, 4, 6 and reduce 0s.
    If < 1: decrease 1, 2, 4, 6 and increase 0s.
    Wickets and others are not touched here.

    A/B exponents and priors/bases are tunable via config/baseline_weights.json
    (see server.py's _compute_league_avg); .get() defaults keep standalone
    callers working.
    """
    prior_bat_sr = league_avg.get('prior_bat_sr', 135.3)
    prior_eco = league_avg.get('prior_eco', 8.38)
    k_bs = league_avg.get('k_balls_sr', 100.0)
    k_be = league_avg.get('k_balls_eco', 60.0)
    bat_a = league_avg.get('bat_power_a', 0.15)
    bat_b = league_avg.get('bat_power_b', 1.2)
    bowl_a = league_avg.get('bowl_power_a', 0.15)
    bowl_b = league_avg.get('bowl_power_b', 0.2)
    bat_power_base = league_avg.get('bat_power_base', 1013.1)
    bowl_power_base = league_avg.get('bowl_power_base', 1.692)
    bat_power_floor = league_avg.get('bat_power_floor', 0.15)
    bowl_power_floor = league_avg.get('bowl_power_floor', 0.15)

    # Batter power: rewards career balls faced (volume) and a high ghost strike rate.
    balls = max(0, batter.career_balls)
    runs = max(0, batter.career_runs)
    ghost_bat_sr = ((runs + k_bs * prior_bat_sr / 100.0) / (balls + k_bs)) * 100.0
    bat_power_raw = (balls ** bat_a) * (ghost_bat_sr ** bat_b)
    bat_power = max(bat_power_floor, bat_power_raw / bat_power_base if bat_power_base > 0 else 1.0)

    # Bowler power: rewards legal balls bowled (volume) and a low ghost economy.
    balls_b = max(0, getattr(bowler, 'legal_balls', 0))
    runs_conceded = balls_b * bowler.eco / 6.0
    ghost_eco = ((runs_conceded + k_be * prior_eco) / (balls_b + k_be)) * 6.0
    bowl_power_raw = (balls_b ** bowl_a) / (ghost_eco ** bowl_b) if ghost_eco > 0 else 0.0
    bowl_power = max(bowl_power_floor, bowl_power_raw / bowl_power_base if bowl_power_base > 0 else 1.0)

    p_str_raw = bat_power / bowl_power

    # Scale vector size to treat this as a secondary nudge (tunable via
    # config/baseline_weights.json's stage2_strike_rate_economy.damp)
    str_damp = league_avg.get('str_damp', 0.2)
    p_str = 1.0 + (p_str_raw - 1.0) * str_damp

    result = {k: float(v) for k, v in weights.items()}
    
    # Store sum of [0, 1, 2, 4, 6] to keep it constant
    active_keys = ['0', '1', '2', '4', '6']
    active_sum = sum(result[k] for k in active_keys)

    # Multiply 1, 2, 4, 6 by p_str
    for k in ['1', '2', '4', '6']:
        result[k] *= p_str

    modified_1246_sum = sum(result[k] for k in ['1', '2', '4', '6'])
    new_0 = active_sum - modified_1246_sum

    if new_0 >= 0:
         result['0'] = new_0
    else:
         result['0'] = 0.0
         # Scale down 1, 2, 4, 6 to conserve active_sum
         if modified_1246_sum > 0:
             scale = active_sum / modified_1246_sum
             for k in ['1', '2', '4', '6']:
                 result[k] *= scale

    # Normalize to exactly 1000.0 units
    total = sum(result.values())
    if total > 0.0:
        result = {k: (v / total) * 1000.0 for k, v in result.items()}
    return result

def apply_stage3_wicket_factor(weights: dict, batter: Batter, bowler: Bowler, league_avg: dict) -> dict:
    """
    Stage 3: The Wicket Factor (P_wkt) -- symmetric longevity + ghost-stat smoothing.

    Both sides use a VOLUME^A x RATE^B score, ghost-stat (Bayesian pseudo-count)
    smoothed toward a league prior so tiny-sample flukes regress to average:
        ghost_sr  = (legal_balls + K_W*prior_sr)  / (wickets    + K_W)   # balls/wkt
        ghost_avg = (runs        + K_D*prior_avg) / (dismissals + K_D)   # runs/out
        bowler_threat   = ( wickets^A_wkt  / ghost_sr^B_wkt  ) / threat_base
        batter_patience = ( runs^A_bat     * ghost_avg^B_bat ) / patience_base
        batter_defense  = 1 / max(patience_floor, batter_patience)
    P_wkt = 1 + (bowler_threat * batter_defense - 1) * damp
    Multiply OUT by P_wkt, adjust everything else proportionally to conserve 1000.0 sum.

    A/B exponents, priors/bases and damp are tunable via
    config/baseline_weights.json (see server.py's _compute_league_avg);
    .get() defaults keep standalone callers working.
    """
    prior_sr = league_avg.get('prior_sr', 21.2)
    prior_avg = league_avg.get('prior_avg', 27.5)
    k_w = league_avg.get('k_wickets', 5.0)
    k_d = league_avg.get('k_dismissals', 4.0)
    threat_a = league_avg.get('threat_a', 0.5)
    threat_b = league_avg.get('threat_b', 1.0)
    patience_a = league_avg.get('patience_a', 1.0 / 3.0)
    patience_b = league_avg.get('patience_b', 1.0)
    threat_base = league_avg.get('threat_base', 0.302)
    patience_base = league_avg.get('patience_base', 315.0)
    patience_floor = league_avg.get('patience_floor', 0.15)

    # Bowler threat: rewards wickets (volume) and a low ghost strike rate.
    wkt = max(0, bowler.wkt)
    balls_b = max(0, getattr(bowler, 'legal_balls', 0))
    if balls_b > 0:
        ghost_sr = (balls_b + k_w * prior_sr) / (wkt + k_w)
    else:
        # Unknown workload (e.g. a legacy roster without legal_balls): fall back
        # to the league prior strike rate rather than a degenerate tiny value.
        ghost_sr = prior_sr
    threat_raw = (wkt ** threat_a) / (ghost_sr ** threat_b) if ghost_sr > 0 else 0.0
    bowler_threat = threat_raw / threat_base if threat_base > 0 else 1.0

    # Batter patience: rewards runs (volume) and a high ghost average.
    runs = max(0, batter.career_runs)
    dismissals = max(0, batter.dismissals)
    ghost_avg = (runs + k_d * prior_avg) / (dismissals + k_d)
    patience_raw = (runs ** patience_a) * (ghost_avg ** patience_b)
    batter_patience = patience_raw / patience_base if patience_base > 0 else 1.0
    batter_defense = 1.0 / max(patience_floor, batter_patience)

    p_wkt_raw = bowler_threat * batter_defense

    # Scale vector size to treat this as a secondary nudge (the single calibration
    # dial; tunable via config/baseline_weights.json's stage3_wicket_factor.damp).
    damp = league_avg.get('wicket_damp', 0.2)
    p_wkt = 1.0 + (p_wkt_raw - 1.0) * damp

    result = {k: float(v) for k, v in weights.items()}
    
    u_out_old = result['Out']
    u_out_new = u_out_old * p_wkt

    # Constraints: Out cannot occupy 100% of the space
    if u_out_new >= 995.0:
        u_out_new = 995.0

    result['Out'] = u_out_new
    delta = u_out_new - u_out_old

    # Total sum of other components
    others_keys = [k for k in result.keys() if k != 'Out']
    others_sum_old = sum(result[k] for k in others_keys)

    if others_sum_old > 0:
        scale = (others_sum_old - delta) / others_sum_old
        for k in others_keys:
            result[k] = max(0.0, result[k] * scale)
    else:
        # If there are no other components (should not happen), distribute the remainder
        remainder = 1000.0 - u_out_new
        for k in others_keys:
            result[k] = remainder / len(others_keys)

    # Normalize to exactly 1000.0 units
    total = sum(result.values())
    if total > 0.0:
        result = {k: (v / total) * 1000.0 for k, v in result.items()}
    return result