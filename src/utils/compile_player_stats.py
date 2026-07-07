# src/utils/compile_player_stats.py

import os
import json
import glob
import time

MATCH_DATA_PATH = "data/All_Matches_Json/*.json"
OUTPUT_FILE = "data/players_historical.json"

BOWLER_CREDITED_DISMISSALS = {
    "bowled",
    "caught",
    "caught and bowled",
    "lbw",
    "stumped",
    "hit wicket"
}

KNOWN_FOREIGNERS = {
    "DA Warner", "SR Watson", "AB de Villiers", "CH Gayle", "AD Russell", "SP Narine", "KA Pollard", "SL Malinga", 
    "JC Buttler", "Q de Kock", "F du Plessis", "Rashid Khan", "TA Boult", "KS Williamson", "K Rabada", "A Nortje", 
    "GJ Maxwell", "SPD Smith", "MEK Hussey", "ML Hayden", "AC Gilchrist", "DJ Bravo", "MP Stoinis", "C Green",
    "TH David", "MR Marsh", "JM Bairstow", "N Pooran", "SO Hetmyer", "R Powell", "AS Joseph", "JO Holder", 
    "LMP Simmons", "DR Smith", "S Badree", "FH Edwards", "R Rampaul", "E Lewis", "DA Miller", "CH Morris", 
    "L Ngidi", "M Morkel", "JA Morkel", "DW Steyn", "GC Smith", "HM Amla", "RR Rossouw", "Imran Tahir", 
    "JP Duminy", "T Bavuma", "M Jansen", "T Stubbs", "D Brevis", "BA Stokes", "MM Ali", "SC Curran", "SM Curran", 
    "EJG Morgan", "KP Pietersen", "JJ Roy", "AD Hales", "DJ Malan", "CJ Jordan", "TS Mills", "MA Wood", 
    "LE Plunkett", "H Brook", "PD Salt", "WG Jacks", "TG Southee", "BB McCullum", "LRPL Taylor", "C Munro", 
    "MJ Santner", "LH Ferguson", "AF Milne", "MJ McClenaghan", "JEC Franklin", "CJ Anderson", "DP Conway", 
    "R Ravindra", "DJ Mitchell", "KC Sangakkara", "DPMD Jayawardene", "TM Dilshan", "PWH de Silva", "M Theekshana", 
    "M Pathirana", "P Nissanka", "B Fernando", "M Muralitharan", "NLTC Perera", "AD Mathews", "KMDN Kulasekara", 
    "Mohammad Nabi", "Mujeeb Ur Rahman", "Fazalhaq Farooqi", "Naveen-ul-Haq", "Rahmanullah Gurbaz", "Noor Ahmad", 
    "Azmatullah Omarzai", "Shakib Al Hasan", "Mustafizur Rahman", "Litton Das", "Mushfiqur Rahim", "Sikandar Raza", 
    "D Wiese", "RN ten Doeschate", "Shoaib Akhtar", "Sohail Tanvir", "Shahid Afridi", "Salman Butt", 
    "Mohammad Hafeez", "Umar Gul", "Kamran Akmal", "Misbah-ul-Haq", "Younis Khan", "AJ Finch", "SE Marsh", 
    "CA Lynn", "PJ Cummins", "MA Starc", "JR Hazlewood", "A Zampa", "B Lee", "SW Tait", "MG Johnson", 
    "JP Faulkner", "DT Christian", "JA Richardson", "JH Kallis", "A Symonds", "DJ Hussey", "ST Jayasuriya",
    "MC Henriques", "CJ McKay", "RJ Harris", "BMAJ Mendis", "H Klaasen"
}

KNOWN_KEEPERS = {
    "MS Dhoni", "Q de Kock", "JC Buttler", "RR Pant", "KL Rahul", "Ishan Kishan", "WP Saha", "KD Karthik", 
    "SV Samson", "JM Bairstow", "N Pooran", "PA Patel", "RV Uthappa", "NV Ojha", "AC Gilchrist", "KC Sangakkara",
    "BB McCullum", "Kamran Akmal", "Litton Das", "Mushfiqur Rahim", "Rahmanullah Gurbaz", "H Klaasen", 
    "Jitesh Sharma", "Dhruv Jurel", "KS Bharat", "Anuj Rawat", "SW Billings", "SS Goswami", "CM Gautam", "MS Bisla"
}

def compile_stats():
    print("Starting player stats compilation with OVR scaling...")
    start_time = time.time()
    
    players = {}
    dynamic_keepers = set()
    
    match_files = glob.glob(MATCH_DATA_PATH)
    total_files = len(match_files)
    print(f"Found {total_files} match files to process.")
    
    processed_count = 0
    for file_path in match_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                match = json.load(f)
                
            info = match.get("info", {})
            players_in_match = info.get("players", {})
            for team, roster in players_in_match.items():
                for name in roster:
                    if name not in players:
                        players[name] = {
                            "name": name,
                            "batting": {"runs": 0, "balls": 0, "fours": 0, "sixes": 0, "dismissals": 0},
                            "bowling": {"runs_conceded": 0, "legal_balls": 0, "wickets": 0}
                        }
            
            for inning in match.get("innings", []):
                for over_data in inning.get("overs", []):
                    for delivery in over_data.get("deliveries", []):
                        batter = delivery.get("batter")
                        bowler = delivery.get("bowler")
                        
                        for p in [batter, bowler]:
                            if p and p not in players:
                                players[p] = {
                                    "name": p,
                                    "batting": {"runs": 0, "balls": 0, "fours": 0, "sixes": 0, "dismissals": 0},
                                    "bowling": {"runs_conceded": 0, "legal_balls": 0, "wickets": 0}
                                }
                                
                        runs_data = delivery.get("runs", {})
                        batter_runs = runs_data.get("batter", 0)
                        
                        extras_data = delivery.get("extras", {})
                        wides = extras_data.get("wides", 0)
                        noballs = extras_data.get("noballs", 0)
                        
                        if wides == 0:
                            players[batter]["batting"]["balls"] += 1
                        
                        players[batter]["batting"]["runs"] += batter_runs
                        if batter_runs == 4:
                            players[batter]["batting"]["fours"] += 1
                        elif batter_runs == 6:
                            players[batter]["batting"]["sixes"] += 1
                            
                        bowler_runs_conceded = batter_runs + wides + noballs
                        players[bowler]["bowling"]["runs_conceded"] += bowler_runs_conceded
                        
                        if wides == 0 and noballs == 0:
                            players[bowler]["bowling"]["legal_balls"] += 1
                            
                        for wicket in delivery.get("wickets", []):
                            player_out = wicket.get("player_out")
                            dismissal_kind = wicket.get("kind", "")
                            
                            if player_out == batter:
                                players[batter]["batting"]["dismissals"] += 1
                            elif player_out in players:
                                players[player_out]["batting"]["dismissals"] += 1
                                
                            if dismissal_kind in BOWLER_CREDITED_DISMISSALS:
                                players[bowler]["bowling"]["wickets"] += 1
                                
                            if dismissal_kind == "stumped":
                                for fielder in wicket.get("fielders", []):
                                    dynamic_keepers.add(fielder.get("name"))
                                
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
            
        processed_count += 1
        if processed_count % 200 == 0:
            print(f"Processed {processed_count}/{total_files} matches...")

    # Phase 2: Compute stats and raw powers
    compiled_items = []
    max_bat_power = 0.0
    max_bowl_power = 0.0
    best_batter = ""
    best_bowler = ""
    
    for name, data in players.items():
        bat = data["batting"]
        bowl = data["bowling"]
        
        runs = bat["runs"]
        balls = bat["balls"]
        dismissals = bat["dismissals"]
        
        bat_sr = float((runs / balls) * 100.0) if balls > 0 else 0.0
        bat_avg = float(runs / dismissals) if dismissals > 0 else float(runs)
        
        # Bayesian Smooth Quality:
        # Pulls tiny samples towards League Medians (Avg 25, SR 130)
        smoothed_avg = (runs + 100.0) / (dismissals + 4.0)
        smoothed_sr = ((runs + 130.0) / (balls + 100.0)) * 100.0
        
        # Batting Power = CubeRoot(Runs) * Smoothed Average * (Smoothed Strike Rate / 100)
        bat_volume = runs ** (1.0/3.0) if runs > 0 else 0.0
        bat_power = bat_volume * smoothed_avg * (smoothed_sr / 100.0)
        
        if bat_power > max_bat_power:
            max_bat_power = bat_power
            best_batter = name
            
        runs_c = bowl["runs_conceded"]
        balls_b = bowl["legal_balls"]
        wickets = bowl["wickets"]
        
        bowl_eco = float((runs_c / balls_b) * 6.0) if balls_b > 0 else 0.0
        bowl_avg = float(runs_c / wickets) if wickets > 0 else 0.0
        bowl_sr = float(balls_b / wickets) if wickets > 0 else 0.0
        
        # Bayesian Smooth Quality for Bowlers:
        # Pulls tiny samples towards League Medians (Eco 8.0, Avg 25)
        smoothed_eco = ((runs_c + 160.0) / (balls_b + 120.0)) * 6.0
        smoothed_bowl_avg = (runs_c + 125.0) / (wickets + 5.0)
        
        # Bowling Power = CubeRoot(Wickets) * (1000 / (Smoothed Eco * Smoothed Average))
        bowl_volume = wickets ** (1.0/3.0) if wickets > 0 else 0.0
        
        bowl_power = bowl_volume * (1000.0 / (smoothed_eco * smoothed_bowl_avg))
        
        if bowl_power > max_bowl_power:
            max_bowl_power = bowl_power
            best_bowler = name
            
        compiled_items.append({
            "name": name,
            "is_keeper": (name in KNOWN_KEEPERS) or (name in dynamic_keepers),
            "is_foreigner": name in KNOWN_FOREIGNERS,
            "batting": {
                "runs": runs,
                "balls": balls,
                "fours": bat["fours"],
                "sixes": bat["sixes"],
                "dismissals": dismissals,
                "avg": round(bat_avg, 2),
                "sr": round(bat_sr, 2)
            },
            "bowling": {
                "runs_conceded": runs_c,
                "legal_balls": balls_b,
                "wickets": wickets,
                "eco": round(bowl_eco, 2),
                "avg": round(bowl_avg, 2),
                "sr": round(bowl_sr, 2)
            },
            "raw_bat_power": bat_power,
            "raw_bowl_power": bowl_power
        })
        
    # Phase 3: Scale powers to 90 OVR (Baseline 55) and format output
    final_players = []
    for p in compiled_items:
        bat_power = p.pop("raw_bat_power")
        bowl_power = p.pop("raw_bowl_power")
        
        b_ovr = int(round(55.0 + (bat_power / max_bat_power) * 44.0)) if max_bat_power > 0 else 55
        bw_ovr = int(round(55.0 + (bowl_power / max_bowl_power) * 44.0)) if max_bowl_power > 0 else 55
        
        b_ovr_capped = max(55, min(99, b_ovr))
        bw_ovr_capped = max(55, min(99, bw_ovr))
        
        p["batting_ovr"] = b_ovr_capped
        p["bowling_ovr"] = bw_ovr_capped
        p["batting"]["ovr"] = b_ovr_capped
        p["bowling"]["ovr"] = bw_ovr_capped
        
        final_players.append(p)
        
    # Sort alphabetically
    final_players.sort(key=lambda x: x["name"])
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    total_players = len(final_players)
    json_str = json.dumps(final_players, indent=2)
    header = (
        f"// Total Players: {total_players}\n"
        f"// Highest Batting OVR (99): {best_batter} (Raw Power: {max_bat_power:.2f})\n"
        f"// Highest Bowling OVR (99): {best_bowler} (Raw Power: {max_bowl_power:.2f})\n"
    )
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(header + json_str)
        
    print(f"Compilation finished! Compiled {total_players} players in {time.time() - start_time:.2f} seconds.")
    print(f"Stats Scaling => Best Batter: {best_batter} | Best Bowler: {best_bowler}")

if __name__ == "__main__":
    compile_stats()
