import json
import random

def determine_role(player):
    """Sorts players into 5 clear IPL Auction roles. Bowlers are split into
    Pacer/Spinner (by the `bowling_style` field baked into players_historical.json
    at compile time) so genuine specialist bowlers get two dedicated auction
    roles' worth of pool slots instead of being lumped into one -- All-Rounder
    stays a single role regardless of bowling style."""
    if player.get('is_keeper', False):
        return "Wicket Keeper"

    bat_ovr = player.get('batting_ovr', 55)
    bowl_ovr = player.get('bowling_ovr', 55)

    if bat_ovr >= 60 and bowl_ovr >= 60:
        return "All-Rounder"
    if bat_ovr > bowl_ovr:
        return "Batsman"
    return "Spinner" if player.get('bowling_style') == "Spin" else "Pacer"

# Marquee starts at a different OVR depending on the role, because the OVR
# distributions differ by role and a flat cut empties the thin ones. Anything
# from MID_FLOOR up to the cut is Mid-Level; everything below is Group 3.
#
# Counts clearing each cut on the 2014-2022 pool under the grid-derived OVRs:
#
#     role            n     74    78    80    85
#     Batsman        86     31    21    18     8
#     Wicket Keeper  26     16    12    10     6
#     All-Rounder    33     19    11    11     7
#     Pacer          70     27    17    12     7
#     Spinner        31      6     3     2     1
#
# SPINNER IS THE THIN ONE and needs its own, lower cut. Elite T20 spin is
# genuinely scarce -- only 31 draftable spinners against 70 pacers -- and the
# spell-based bowling rating puts most of the very top on pace, since a wicket is
# worth 5.57 runs and the heaviest strike bowlers are quick. At 80 the tier held
# TWO spinners, so an 8-team table (which draws 8 Marquee lots per role) would
# see the same two men every auction. At 74 it holds 6.
#
# These are absolute numbers against a scale that gets re-derived, so they need
# re-checking whenever ml/train/derive_ovr_grid.py runs. A per-role PERCENTILE
# would not -- worth doing if this needs tuning a third time.
MARQUEE_CUT = {
    "Batsman": 85, "Wicket Keeper": 85, "All-Rounder": 85,
    "Pacer": 80, "Spinner": 74,
}
MID_FLOOR = 70

def determine_tier(ovr, role="Batsman"):
    """Tier for a player of `role` rated `ovr` on that role's own OVR key.

    Judged per role rather than on one global line: `Pacer`/`Spinner` are rated
    on bowling_ovr (see `_role_ovr_key`), which sits ~5 points below batting_ovr
    for equivalent quality, so they get a correspondingly lower Marquee cut.
    """
    if ovr >= MARQUEE_CUT.get(role, 85): return "Marquee"
    if ovr >= MID_FLOOR: return "Mid-Level"
    return "Group 3"

def _role_ovr_key(role):
    """Returns the OVR field a set's role should be ranked/balanced on."""
    if role in ("Pacer", "Spinner"):
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

def auction_pool_size_per_set(num_teams, players_per_team=25, num_sets=15):
    """At least `players_per_team` players available per team in the game,
    spread evenly across the 15 sets (3 tiers x 5 roles).

    Superseded by `auction_set_sizes` for live auctions -- kept because it is a
    reasonable uniform default and the module's __main__ demo still uses it.
    """
    target_total = players_per_team * max(1, num_teams)
    per_set = (target_total + num_sets - 1) // num_sets
    return max(5, per_set)

def auction_set_sizes(num_teams):
    """Players per set, sized *per tier* so the top tier is the scarcest.

    Sizing every set the same gives each tier a third of the lots, which handed
    out 8-12 Marquee lots per team against an XI of 11 -- at two teams there were
    25 stars for 22 XI slots, so they were never contested and the money never
    reached the Mid-Level sets. These ratios instead hold Marquee at ~5 lots per
    team at every table size, comfortably under the 11 an XI needs, which forces
    the other ~6 places to be bought from Mid-Level.
    """
    n = max(1, num_teams)
    return {
        "Marquee": max(2, n),
        "Mid-Level": max(4, -(-9 * n // 5)),   # ceil(1.8 x n)
        "Group 3": max(3, -(-7 * n // 5)),     # ceil(1.4 x n)
    }

def generate_draft_pool(all_players, players_per_set=5, set_sizes=None):
    """
    Generates 15 sets (3 Tiers x 5 Roles: Batsman, Pacer, Spinner, All-Rounder,
    Wicket Keeper).
    Pulls up to `players_per_set` players per set, or -- when `set_sizes` is
    given as a {tier: count} dict from `auction_set_sizes` -- a different count
    per tier, so the Marquee sets can be deliberately smaller than the rest.
    A set is still capped by however many players its bucket actually holds.
    Selection is quality-balanced (drawn from bat/bowl-OVR bands, not pure
    random), and the *presentation order* within a set is shuffled independently
    from whatever order the pool ends up listed in.
    No overseas quota per set -- the XI's own overseas limit handles that, and
    far better; see the note in the selection loop.
    """
    buckets = {
        "Marquee": {"Batsman": [], "Pacer": [], "Spinner": [], "All-Rounder": [], "Wicket Keeper": []},
        "Mid-Level": {"Batsman": [], "Pacer": [], "Spinner": [], "All-Rounder": [], "Wicket Keeper": []},
        "Group 3": {"Batsman": [], "Pacer": [], "Spinner": [], "All-Rounder": [], "Wicket Keeper": []}
    }

    for p in all_players:
        role = determine_role(p)
        # Rated on the role's own OVR key, so a pacer is tiered on his bowling
        # and a batsman on his batting -- not on whichever of the two is higher.
        tier = determine_tier(_role_ovr_key(role)(p), role)
        buckets[tier][role].append(p)

    draft_sets = []
    set_number = 1
    total_players_pulled = 0
    total_foreigners_pulled = 0

    tiers = ["Marquee", "Mid-Level", "Group 3"]
    roles = ["Batsman", "Pacer", "Wicket Keeper", "Spinner", "All-Rounder"]
    sizes = set_sizes or {t: players_per_set for t in tiers}

    for tier in tiers:
        size = max(1, int(sizes.get(tier, players_per_set)))
        for role in roles:
            candidates = _balanced_order(buckets[tier][role], _role_ovr_key(role), size)

            selected_for_set = []

            # No overseas quota on a SET. There used to be one (40%, floor 2),
            # and it starved the sets it was meant to balance: the 2014-2022
            # Marquee Batsman bucket holds 7 players of whom 5 are overseas, so
            # a 5-lot set could only be filled to 4 -- fewer marquee batsmen
            # than there were teams to buy them.
            #
            # It was solving a problem that is already solved downstream, and
            # better: an XI may field at most XI_MAX_OVERSEAS, enforced when the
            # XI is locked. Who you are ALLOWED to buy should not be rationed --
            # overspending on players you then cannot all field is a real and
            # interesting way to lose an auction.
            for p in candidates:
                if len(selected_for_set) >= size:
                    break
                if p.get('is_foreigner', False):
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
