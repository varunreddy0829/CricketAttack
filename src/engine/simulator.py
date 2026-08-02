# src/engine/simulator.py

import random
from src.models.player import Batter, Bowler
from src.models.match_state import MatchState
from src.engine.stats_calculator import (
    BASELINE_WEIGHTS,
    apply_stage1_ovr,
    apply_stage2_strike_rate_economy,
    apply_stage3_wicket_factor
)
from src.engine.intent_handler import apply_intents
from src.engine.conditions import apply_conditions
from src.engine.roles import apply_modes, apply_roles

# Independent extras master probability configuration. Measured from 290,611 real
# IPL deliveries (2008-2026): extras run 3.75% of balls, and wides make up 89% of
# those extras (not a 70/30 split -- no-balls are genuinely rare in comparison).
EXTRAS_PROB = 0.0375  # was 0.04
WIDE_SHARE_OF_EXTRAS = 0.89  # was 0.7; single source of truth -- see server.py

# Wicket cascade: each wicket already fallen in the CURRENT over multiplies the
# remaining balls' Out weight by this factor (stacking: 2 wickets -> x0.20),
# with the freed weight moved to dots -- the new batter blocks, the field
# resets. Resets every over (the count is per-over, owned by the caller).
#
# Tuned over 1500-innings sweeps. Was 0.6; tightened to 0.45 because 3-wicket
# overs still turned up about once every 13 innings, which reads as a collapse
# far too regularly. At 0.45 that halves to once every ~27 innings and
# 4-in-an-over is ~1 per 9000 overs, while the innings shape is essentially
# untouched (avg 149 -> 151 runs, all-out 38.5% -> 37.5%) and ordinary
# 2-wicket overs stay common. Going further (0.4 and below) starts visibly
# suppressing all-out finishes, which flattens run chases.
WICKET_CASCADE_MULT = 0.45

def calculate_single_ball(striker: Batter, bowler: Bowler, league_avg: dict, context: dict = None) -> str:
    """
    Runs the striker and bowler through the math engine pipeline.
    Returns the resolved delivery outcome (e.g. '0', '1', '2', '3', '4', '5', '6', 'Out').
    `context` (optional) carries match conditions — pitch, innings phase,
    gambits (see conditions.apply_conditions); None skips the stage.
    """
    # Stage 1: OVR Strength Adjustment (starts from the shared global baseline --
    # no per-batter DNA stage; see BASELINE_WEIGHTS' comment in stats_calculator.py)
    weights_s1 = apply_stage1_ovr(BASELINE_WEIGHTS, striker, bowler)

    # Stage 2: Strike Rate vs Economy Adjustment
    weights_s2 = apply_stage2_strike_rate_economy(weights_s1, striker, bowler, league_avg)

    # Stage 3: Wicket Factor Adjustment
    weights_s3 = apply_stage3_wicket_factor(weights_s2, striker, bowler, league_avg)

    # Stage 3.5: Match conditions (pitch, phase, gambits)
    weights_c = apply_conditions(weights_s3, context)

    # Stage 4 + 5: roles. Stage 4 (apply_modes) is the raw intent CHOICE —
    # three fixed presets per side that move ALL probabilities including Out,
    # identically for every player (the old slider, reduced to 3 options).
    # Stage 5 (apply_roles) then adds the Playstyle-Grid bonus on top: pure
    # upside for specialists (elite attacker's 4/6 x1.5 from dots, Out
    # untouched; bowler mirror). Legacy callers without roles fall back to
    # the intent meters unchanged.
    if context and context.get("bat_role"):
        weights_m = apply_modes(
            weights_c,
            bat_mode=context.get("bat_role"),
            bowl_mode=context.get("bowl_role"),
        )
        final_weights = apply_roles(
            weights_m,
            bat_role=context.get("bat_role"),
            bat_grid=context.get("bat_grid", 50),
            bowl_role=context.get("bowl_role"),
            bowl_grid=context.get("bowl_grid", 50),
        )
    else:
        final_weights = apply_intents(weights_c, striker.intent, bowler.intent, league_avg)

    # Stage 6: wicket cascade. A pure Out -> '0' transfer, so the 1000.0 sum
    # is conserved by construction. Only active when the caller tracks the
    # per-over wicket count in context (the web server does); legacy callers
    # without it behave exactly as before.
    wkts_this_over = (context or {}).get("wickets_this_over", 0)
    if wkts_this_over > 0:
        factor = WICKET_CASCADE_MULT ** wkts_this_over
        removed = final_weights['Out'] * (1.0 - factor)
        final_weights['Out'] -= removed
        final_weights['0'] += removed

    outcomes = list(final_weights.keys())
    values = list(final_weights.values())
    
    # Normalize probabilities to sum up to exactly 1.0 (for random.choices)
    total_val = sum(values)
    probs = [v / total_val for v in values] if total_val > 0.0 else [1.0 / len(values)] * len(values)
    
    return random.choices(outcomes, weights=probs, k=1)[0]

def simulate_over(state: MatchState, bowler: Bowler, league_avg: dict) -> list:
    """
    Simulates a full 6 legal ball over, updating MatchState.
    Handles strike rotation and wicket consequences.
    Returns a list of commentary strings for the game UI.
    """
    over_commentary = []
    legal_balls = 0
    balls_this_session = 0
    
    # We only bowl maximum 6 valid deliveries in an over. 
    # If the over is partially completed (e.g. 3 balls in and a wicket fell), we bowl the remainder (6 - balls % 6).
    remainder_balls = 6 - (state.balls % 6) if state.balls % 6 != 0 else 6
    if remainder_balls == 0: remainder_balls = 6 # if exactly on the cusp of a new over
    
    while legal_balls < remainder_balls:
        # Check if team is already all out
        if state.is_all_out() or state.get_striker() is None:
            over_commentary.append("PENDING NEW BATSMAN / ALL OUT")
            break
            
        striker = state.get_striker()
        ball_num_str = f"{state.balls // 6}.{(state.balls % 6) + 1}"
        
        # 1. Check for independent extras (Wide or No Ball)
        if random.random() < EXTRAS_PROB:
            extra_type = "Wide" if random.random() < WIDE_SHARE_OF_EXTRAS else "No Ball"
            state.add_extra()
            over_commentary.append(f"[{ball_num_str}] {extra_type}! {striker.name} watches it go. 1 Run.")
            if state.target is not None and state.runs >= state.target:
                over_commentary.append("🎯 TARGET REACHED! INNINGS OVER.")
                break
            # Extras are re-bowled, legal_balls is not incremented, and state.balls is not incremented as a legal delivery.
            continue
            
        # 2. Resovle legal delivery outcome via math engine
        outcome = calculate_single_ball(striker, bowler, league_avg)
        state.add_ball()
        
        if outcome == "Out":
            over_commentary.append(f"[{ball_num_str}] OUT! {striker.name} is clean bowled by {bowler.name}.")
            state.handle_wicket()
            legal_balls += 1
            break # Break mid-over to allow manual batsman selection
        else:
            runs = int(outcome)
            state.add_runs(runs)
            over_commentary.append(f"[{ball_num_str}] {runs} Run(s) to {striker.name}.")
            
            if state.target is not None and state.runs >= state.target:
                over_commentary.append("🎯 TARGET REACHED! INNINGS OVER.")
                legal_balls += 1
                break
            
            # Strike rotation on odd runs runs
            if runs in [1, 3, 5]:
                state.rotate_strike()
                
        legal_balls += 1
        
    # Rotate strike automatically at the end of the over if there are valid batsmen left
    if state.balls > 0 and state.balls % 6 == 0 and not state.is_all_out() and state.get_striker() is not None:
        state.rotate_strike()
        
    return over_commentary