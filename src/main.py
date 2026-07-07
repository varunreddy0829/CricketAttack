# src/main.py

import json
import sys
from src.models.player import Batter, Bowler
from src.models.match_state import MatchState
from src.engine.simulator import simulate_over

def load_roster(filepath: str):
    """
    Loads rosters and league data from a JSON file.
    """
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.", file=sys.stderr)
        sys.exit(1)

def main():
    print("=" * 60)
    print("      CRICKET ATTACK - MATHEMATICAL PROBABILITY ENGINE      ")
    print("=" * 60)

    # 1. Load Data
    data = load_roster("data/rosters_2016.json")
    
    # 2. Parse Players
    batters_db = {b['name']: Batter(**b) for b in data['batters']}
    bowlers_db = {bw['name']: Bowler(**bw) for bw in data['bowlers']}
    league_avg = data['league_avg']

    print("\nAvailable Batters:")
    for name, b in batters_db.items():
        print(f" - {name} (OVR: {b.ovr}, Avg: {b.avg:.1f}, SR: {b.sr:.1f})")

    print("\nAvailable Bowlers:")
    for name, bw in bowlers_db.items():
        print(f" - {name} (OVR: {bw.ovr}, Economy: {bw.eco:.2f}, Wickets: {bw.wkt})")
        
    print("\nLeague Averages:")
    print(f" - League Batting Average: {league_avg['avg_batting_avg']}")
    print(f" - League Season Wickets per Bowler: {league_avg['avg_wickets']}")

    # 3. Setup Lineup
    # We will simulate Kohli and Russell opening the innings, with Kohli on strike initially.
    striker_name = "Virat Kohli"
    non_striker_name = "Andre Russell"
    
    striker = batters_db[striker_name]
    non_striker = batters_db[non_striker_name]
    lineup = [striker, non_striker]
    
    state = MatchState(lineup)

    # Choose bowler
    print("\n" + "-" * 50)
    print("Select Bowler to Sim an Over against Kohli & Russell:")
    print("1. Jasprit Bumrah (Elite Bowler)")
    print("2. Terrible Parttimer (Vulnerable Bowler)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    if choice == "2":
        selected_bowler = bowlers_db["Terrible Parttimer"]
    else:
        selected_bowler = bowlers_db["Jasprit Bumrah"]

    # Input intents
    print("\n" + "-" * 50)
    print("Set Player Strategic Intent Meters (0 = Defensive, 50 = Balanced, 100 = Attacking)")
    
    def get_intent_input(prompt_msg: str) -> int:
        while True:
            try:
                val = input(prompt_msg).strip()
                if val == "":
                    return 50
                val_int = int(val)
                if 0 <= val_int <= 100:
                    return val_int
                print("Intent must be between 0 and 100.")
            except ValueError:
                print("Please enter a valid integer.")

    striker.intent = get_intent_input(f"Enter Strike Batter ({striker.name}) Intent [0-100] (default 50): ")
    non_striker.intent = get_intent_input(f"Enter Off-Strike Batter ({non_striker.name}) Intent [0-100] (default 50): ")
    selected_bowler.intent = get_intent_input(f"Enter Bowler ({selected_bowler.name}) Intent [0-100] (default 50): ")

    print("\n" + "=" * 50)
    print(f"Simulating Over: {selected_bowler.name} bowling to {striker.name} (Strike) & {non_striker.name}")
    print(f"Intents: {striker.name} ({striker.intent}) | {non_striker.name} ({non_striker.intent}) | {selected_bowler.name} ({selected_bowler.intent})")
    print("=" * 50)

    # Run the over simulation
    commentary = simulate_over(state, selected_bowler, league_avg)

    # Print ball-by-ball commentary
    for line in commentary:
        print(line)

    print("\n" + "=" * 50)
    print("                  FINAL SCORECARD                  ")
    print("=" * 50)
    print(f"Runs: {state.runs}")
    print(f"Wickets lost: {state.wickets}")
    print(f"Extras conceded: {state.extras}")
    print(f"Legal Deliveries faced: {state.balls}")
    print(f"Overs: {state.balls // 6}.{state.balls % 6}")
    
    # Print status of batsmen
    print("\nBatting Status:")
    s = state.get_striker()
    ns = state.get_non_striker()
    s_name = s.name if s else "Dismissed"
    ns_name = ns.name if ns else "Dismissed"
    print(f" - On Strike: {s_name}")
    print(f" - Off Strike: {ns_name}")
    print("=" * 50)

if __name__ == "__main__":
    main()
