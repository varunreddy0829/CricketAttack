# src/engine/intent_handler.py

def apply_single_intent(weights: dict, intent_val: float, strength: float = 1.0) -> dict:
    """
    Applies a single numeric player intent (0 to 100) to the outcomes.
    50 is balanced.
    Values > 50 increase runs (1, 2, 4, 6) and wicket probability (Out), and decrease 0.
    Values < 50 decrease runs (1, 2, 4, 6) and wicket probability (Out), and increase 0.

    `strength` scales how far the slider reaches at the extremes (tunable via
    config/baseline_weights.json's stage4_intent.strength). At the default
    strength=1.0, the total swing is +-25%; at strength=2.0 (the live default),
    dot-ball probability hits exactly 0% at full aggression -- see the config
    file's comment for why 2.0 was chosen over going higher.
    """
    delta = intent_val - 50.0
    p = (delta / 2.0) * strength  # Percentage change (e.g., +20 intent -> +10% * strength multiplier change)
    
    result = {k: float(v) for k, v in weights.items()}
    
    # 3s and 5s are static rare events (total 5.0 units out of 1000)
    static_keys = ['3', '5']
    mod_keys = ['0', '1', '2', '4', '6', 'Out']
    
    # Apply multipliers
    for k in ['1', '2', '4', '6', 'Out']:
        result[k] *= (1.0 + p / 100.0)
        
    result['0'] *= (1.0 - 2.0 * p / 100.0)
    
    # Clip to non-negative values
    for k in mod_keys:
        result[k] = max(0.0, result[k])
        
    # Conserve sum of mod_keys to 995.0 (since 3s and 5s are locked to 4.0 and 1.0)
    mod_sum = sum(result[k] for k in mod_keys)
    if mod_sum > 0.0:
        scale = 995.0 / mod_sum
        for k in mod_keys:
            result[k] *= scale
    else:
        # Fallback if everything became zero
        for k in mod_keys:
            result[k] = 995.0 / len(mod_keys)
            
    # Set static constants back
    result['3'] = 4.0
    result['5'] = 1.0
    
    return result

def apply_intents(weights: dict, striker_intent: int, bowler_intent: int, league_avg: dict = None) -> dict:
    """
    Applies Stage 4 human intent multipliers for both batter and bowler to the probability weights.
    Both intents are applied sequentially, conserving the total sum of 1000.

    league_avg carries intent_strength (server computes it from config/baseline_weights.json);
    omitting it (standalone callers) keeps the original strength=1.0 behavior.
    """
    strength = league_avg.get('intent_strength', 1.0) if league_avg else 1.0
    weights = apply_single_intent(weights, float(striker_intent), strength)
    weights = apply_single_intent(weights, float(bowler_intent), strength)
    return weights