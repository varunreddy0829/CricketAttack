import json
import random

def determine_role(player):
    """Sorts players into 4 clear IPL Auction roles."""
    if player.get('is_keeper', False):
        return "Wicket Keeper"
        
    bat_ovr = player.get('batting_ovr', 55)
    bowl_ovr = player.get('bowling_ovr', 55)
    
    if bat_ovr >= 60 and bowl_ovr >= 60:
        return "All-Rounder"
    if bat_ovr > bowl_ovr:
        return "Batsman"
    return "Bowler"

def determine_tier(max_ovr):
    if max_ovr >= 75: return "Marquee"
    if max_ovr >= 65: return "Mid-Level"
    return "Group 3"

def generate_draft_pool(all_players):
    """
    Generates 12 sets (3 Tiers x 4 Roles). 
    Pulls exactly 5 random players per set to keep the Draft Pool at 60 active players.
    Max 2 foreigners allowed per set (ensuring conservative <40% overseas ratio globally).
    """
    buckets = {
        "Marquee": {"Batsman": [], "Bowler": [], "All-Rounder": [], "Wicket Keeper": []},
        "Mid-Level": {"Batsman": [], "Bowler": [], "All-Rounder": [], "Wicket Keeper": []},
        "Group 3": {"Batsman": [], "Bowler": [], "All-Rounder": [], "Wicket Keeper": []}
    }
    
    for p in all_players:
        role = determine_role(p)
        max_ovr = max(p.get('batting_ovr', 55), p.get('bowling_ovr', 55))
        tier = determine_tier(max_ovr)
        buckets[tier][role].append(p)
        
    draft_sets = []
    set_number = 1
    total_players_pulled = 0
    total_foreigners_pulled = 0
    
    tiers = ["Marquee", "Mid-Level", "Group 3"]
    roles = ["Batsman", "Bowler", "All-Rounder", "Wicket Keeper"]
    
    for tier in tiers:
        for role in roles:
            available = buckets[tier][role]
            random.shuffle(available)
            
            selected_for_set = []
            foreigner_count_in_set = 0
            
            for p in available:
                if len(selected_for_set) >= 5: # Strict 5 Players per Set
                    break
                    
                is_for = p.get('is_foreigner', False)
                # Hard limit: Max 2 Foreigners per Set of 5 (40%)
                if is_for and foreigner_count_in_set >= 2:
                    continue
                
                if is_for: 
                    foreigner_count_in_set += 1
                    total_foreigners_pulled += 1
                    
                selected_for_set.append(p)
                total_players_pulled += 1
                
            draft_sets.append({
                "set_id": set_number,
                "tier": tier,
                "role": role,
                "players": selected_for_set
            })
            set_number += 1
            
    return draft_sets, total_players_pulled, total_foreigners_pulled

if __name__ == "__main__":
    HISTORICAL_PATH = 'data/players_historical.json'
    
    with open(HISTORICAL_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
        historical_data = json.loads(text[text.find('['):])
        
    print("Generating randomized Draft Pool from 811 Player Databank...")
    random.seed() # Use random seed for different pulls every time
    draft, count, overseas_count = generate_draft_pool(historical_data)
    
    print(f"\n--- SUCCESS ---")
    print(f"Generated {len(draft)} Sets. Total Players: {count}")
    print(f"Total Overseas (Foreigners): {overseas_count} ({(overseas_count/count)*100:.1f}%)")
    
    print("\n--- SAMPLE VIEW: SET 1 ---")
    set1 = draft[0]
    print(f"Set ID: {set1['set_id']} | Level: {set1['tier']} | Role: {set1['role']}")
    for p in set1['players']:
        flag = "[Overseas]" if p.get('is_foreigner') else "[Local]"
        print(f" - {p['name']} {flag} (Bat: {p.get('batting_ovr')}, Bowl: {p.get('bowling_ovr')})")
        
    print("\n--- SAMPLE VIEW: SET 12 ---")
    set12 = draft[-1]
    print(f"Set ID: {set12['set_id']} | Level: {set12['tier']} | Role: {set12['role']}")
    for p in set12['players']:
        flag = "[Overseas]" if p.get('is_foreigner') else "[Local]"
        print(f" - {p['name']} {flag} (Bat: {p.get('batting_ovr')}, Bowl: {p.get('bowling_ovr')})")
