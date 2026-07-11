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

def _role_ovr_key(role):
    """Returns the OVR field a set's role should be ranked/balanced on."""
    if role == "Bowler":
        return lambda p: p.get('bowling_ovr', 55)
    if role == "Batsman":
        return lambda p: p.get('batting_ovr', 55)
    return lambda p: max(p.get('batting_ovr', 55), p.get('bowling_ovr', 55))

def _balanced_order(players, key_fn, num_bands):
    """Orders players into quality bands (top/upper/mid/lower/weak by `key_fn`),
    shuffles within each band, then round-robins across bands. Selecting the
    first N of the result yields a set whose average OVR tracks the bucket's
    own average instead of drifting strong/weak on pure random luck, while
    still being randomized (not deterministic top-N or list order)."""
    if not players:
        return []
    ordered = sorted(players, key=key_fn, reverse=True)
    num_bands = max(1, min(num_bands, len(ordered)))
    band_size = (len(ordered) + num_bands - 1) // num_bands
    bands = [ordered[i:i + band_size] for i in range(0, len(ordered), band_size)]
    for b in bands:
        random.shuffle(b)
    result = []
    max_len = max(len(b) for b in bands)
    for i in range(max_len):
        for b in bands:
            if i < len(b):
                result.append(b[i])
    return result

def auction_pool_size_per_set(num_teams, players_per_team=25, num_sets=12):
    """At least `players_per_team` players available per team in the game,
    spread evenly across the 12 sets."""
    target_total = players_per_team * max(1, num_teams)
    per_set = (target_total + num_sets - 1) // num_sets
    return max(5, per_set)

def generate_draft_pool(all_players, players_per_set=5):
    """
    Generates 12 sets (3 Tiers x 4 Roles).
    Pulls up to `players_per_set` players per set (default 5, scaled up by
    callers for larger tournaments so there's always at least ~25 players
    per team). Selection is quality-balanced (drawn from bat/bowl-OVR bands,
    not pure random), and the *presentation order* within a set is shuffled
    independently from whatever order the pool ends up listed in.
    Max 40% foreigners per set (ensuring conservative overseas ratio globally).
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
    max_foreigners_per_set = max(2, round(players_per_set * 0.4))

    for tier in tiers:
        for role in roles:
            candidates = _balanced_order(buckets[tier][role], _role_ovr_key(role), players_per_set)

            selected_for_set = []
            foreigner_count_in_set = 0

            for p in candidates:
                if len(selected_for_set) >= players_per_set:
                    break

                is_for = p.get('is_foreigner', False)
                if is_for and foreigner_count_in_set >= max_foreigners_per_set:
                    continue

                if is_for:
                    foreigner_count_in_set += 1
                    total_foreigners_pulled += 1

                selected_for_set.append(p)
                total_players_pulled += 1

            # Presentation order (the order players come up for bidding) is
            # shuffled independently of the balanced selection order above,
            # so it doesn't just mirror a predictable pattern.
            random.shuffle(selected_for_set)

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
