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

# Independent extras master probability configuration
EXTRAS_PROB = 0.04  # 4% chance of an extra (Wide or No Ball)

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

    # Stage 4: Intent Meter Adjustment (always last)
    final_weights = apply_intents(weights_c, striker.intent, bowler.intent, league_avg)
    
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
            extra_type = "Wide" if random.random() < 0.7 else "No Ball"
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