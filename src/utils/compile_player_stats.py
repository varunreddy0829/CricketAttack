# src/utils/compile_player_stats.py

import os
import json
import glob
import time
import math
import bisect
import statistics

MATCH_DATA_PATH = "data/All_Matches_Json/*.json"
OUTPUT_FILE = "data/players_historical.json"

# Innings phases by 0-indexed over number (T20): 0-5 powerplay, 6-14 middle,
# 15-19 death. Drives the per-phase style-fit tables (see compute_style_fits).
PHASES = ("pp", "mid", "death")

def _phase_of(over_idx):
    if over_idx <= 5:
        return "pp"
    if over_idx <= 14:
        return "mid"
    return "death"

def _blank_phase():
    return {"balls": 0, "runs": 0, "fours": 0, "sixes": 0,
            "ones": 0, "twos": 0, "threes": 0, "dots": 0, "dismissals": 0}

def _blank_bowl_phase():
    # per-phase bowling ledger for the bowler style-fit grid:
    #   balls = legal balls bowled, wkts = bowler-credited wickets,
    #   br = boundary runs conceded (4s+6s off the bat),
    #   rr = running runs conceded (1s+2s+3s off the bat)
    return {"balls": 0, "wkts": 0, "br": 0, "rr": 0}

STYLES = ("attack", "anchor", "rotate")          # batting grid cells (anchor == defend)
BOWL_STYLES = ("attack", "contain", "defend")    # bowling grid cells
FIT_PRIOR_BALLS = 40.0    # ghost-stat strength: every player pre-loaded with 40 league-average balls
FIT_QUALIFY_BALLS = 300   # career balls (faced / bowled) to count as an established reference
# The grid score is  volume^E * rate,  with E derived so longevity carries a
# chosen SHARE of the ranking:  E = (w/(1-w)) / s,  where s is each cell's own
# measured spread ratio SD(log volume)/SD(log rate). w = 0.70 => 70% longevity,
# 30% rate (locked). See compute_style_fits / compute_bowler_fits.
FIT_LONGEVITY_SHARE = 0.70

# A player only earns a role label if some fit cell clears this bar -- no
# fallback label for someone who clears nothing (a pure bowler should show
# no batting flavor badge at all, not a nonsense one from a near-zero cell).
# If multiple cells clear it, only the single best-scoring one is kept as the
# displayed label (see compute_style_fits) -- these are flavor badges for
# viewers, not a stat. The fit CELLS themselves are the raw 0-99
# percentile-vs-all-players numbers and are never rescaled or removed.
ROLE_THRESHOLD = 70

# (label, style, phase) -> a player qualifies for `label` if fit[phase][style]
# >= ROLE_THRESHOLD. Attack is phase-specific (3 labels); anchor/rotate use a
# player's best phase (one label each).
ROLE_DEFS = [
    ("Powerplay Basher", "attack", "pp"),
    ("Middle Enforcer", "attack", "mid"),
    ("Finisher", "attack", "death"),
    ("Anchor", "anchor", None),        # None -> best over all phases
    ("Accumulator", "rotate", None),
]

def _exp_from_spread(vols, rates):
    """Volume exponent E = (w/(1-w)) / s, where w = FIT_LONGEVITY_SHARE and
    s = SD(log volume) / SD(log rate) is the cell's own measured spread ratio.
    This makes longevity carry exactly a `w` share of the ranking. Falls back to
    0.30 on a too-thin or degenerate sample."""
    lv = [math.log(v) for v in vols if v > 0]
    lr = [math.log(r) for r in rates if r > 0]
    if len(lv) < 2 or len(lr) < 2:
        return 0.30
    spread_rate = statistics.pstdev(lr)
    if spread_rate <= 0.0:
        return 0.30
    s = statistics.pstdev(lv) / spread_rate
    if s <= 0.0:
        return 0.30
    w = FIT_LONGEVITY_SHARE
    return (w / (1.0 - w)) / s

def _percentiles(scores, ref_names):
    """Percentile-rank every player's score (0-99, 50 = median) against the
    reference pool `ref_names`."""
    arr = sorted(scores[n] for n in ref_names)
    def pct(v):
        if not arr:
            return 50
        return max(1, min(99, round(bisect.bisect_right(arr, v) / len(arr) * 100)))
    return {n: pct(scores[n]) for n in scores}

def compute_style_fits(players):
    """Attach a per-player 'style_fit' 3x3 table -- {attack, anchor, rotate} x
    {pp, mid, death}, each 0-99 -- plus multi-role labels. Locked recipe, same
    machine for every cell:
        score = volume^E * ghost-smoothed rate   (E gives 70% longevity / 30% rate)
        grid  = percentile of score vs established batsmen (>=300 career balls)
      - attack: rate = boundary runs/ball (4s/6s), volume = total runs
      - rotate: rate = running runs/ball (1s/2s/3s), volume = running runs only
      - anchor: rate = balls per dismissal, volume = balls faced   (== the defend role)
    Ghost stats = every player pre-loaded with FIT_PRIOR_BALLS league-average
    balls, so tiny samples land near the median.
    """
    K = FIT_PRIOR_BALLS
    league = {}
    for ph in PHASES:
        tb = tbr = trr = td = 0
        for p in players.values():
            d = p["bat_phase"][ph]
            tb += d["balls"]
            tbr += 4 * d["fours"] + 6 * d["sixes"]
            trr += d["ones"] + 2 * d["twos"] + 3 * d["threes"]
            td += d["dismissals"]
        league[ph] = {
            "attack": (tbr / tb) if tb else 0.0,
            "rotate": (trr / tb) if tb else 0.0,
            "bpd": (tb / td) if td else 25.0,
        }

    def rate_vol(d, ph, style):
        b, dis = d["balls"], d["dismissals"]
        br = 4 * d["fours"] + 6 * d["sixes"]
        rr = d["ones"] + 2 * d["twos"] + 3 * d["threes"]
        lg = league[ph]
        if style == "attack":
            return (br + K * lg["attack"]) / (b + K), float(d["runs"])
        if style == "rotate":
            return (rr + K * lg["rotate"]) / (b + K), float(rr)
        # anchor / defend
        return (b + K) / (dis + K / lg["bpd"]), float(b)

    qual = [n for n, p in players.items() if p["batting"]["balls"] >= FIT_QUALIFY_BALLS]

    fit = {n: {ph: {} for ph in PHASES} for n in players}
    for ph in PHASES:
        for s in STYLES:
            rates, vols = {}, {}
            for n, p in players.items():
                r, v = rate_vol(p["bat_phase"][ph], ph, s)
                rates[n], vols[n] = r, v
            ref_names = [n for n in qual if players[n]["bat_phase"][ph]["balls"] > 0]
            E = _exp_from_spread([vols[n] for n in ref_names], [rates[n] for n in ref_names])
            scores = {n: (vols[n] ** E) * rates[n] if vols[n] > 0 else 0.0 for n in players}
            grid = _percentiles(scores, ref_names)
            for n in players:
                fit[n][ph][s] = grid[n]

    # attach fit + a single flavor-badge label. A player only gets a label at
    # all if some cell genuinely clears ROLE_THRESHOLD -- there is NO
    # fallback-to-best-available-cell anymore. That fallback used to force a
    # label onto everyone (e.g. a pure bowler's near-zero powerplay-attack
    # score of 1 still "won" as the least-bad cell and got labelled Powerplay
    # Basher), which was actively misleading. If several cells clear 70, only
    # the single highest-scoring one is kept -- these are flavor badges for
    # viewers, not a stat, so one clean label beats a pile of qualifiers.
    for name, p in players.items():
        f = fit[name]
        p["style_fit"] = f
        p["signature"] = {s: max(PHASES, key=lambda ph: f[ph][s]) for s in STYLES}
        roles = []
        for label, style, phase in ROLE_DEFS:
            score = f[phase][style] if phase else max(f[ph][style] for ph in PHASES)
            if score >= ROLE_THRESHOLD:
                roles.append({"label": label, "score": score})
        roles.sort(key=lambda r: -r["score"])
        roles = roles[:1]
        p["roles"] = roles
        p["role"] = roles[0]["label"] if roles else None

def compute_bowler_fits(players):
    """Attach a per-player 'bowl_fit' 3x3 table -- {attack, contain, defend} x
    {pp, mid, death}, each 0-99 -- the mirror of the batting grid read from the
    bowling ledger. Same machine (score = volume^E * rate, 70% longevity,
    percentile vs established bowlers >=300 career legal balls):
      - attack : rate = wickets/ball,                    volume = wickets (higher = better)
      - contain: rate = 1 / (running runs conceded/ball), volume = balls  (tighter = better)
      - defend : rate = 1 / (boundary runs conceded/ball), volume = balls (stingier = better)
    """
    K = FIT_PRIOR_BALLS
    league = {}
    for ph in PHASES:
        tb = tw = tbr = trr = 0
        for p in players.values():
            d = p["bowl_phase"][ph]
            tb += d["balls"]; tw += d["wkts"]; tbr += d["br"]; trr += d["rr"]
        league[ph] = {
            "wr": (tw / tb) if tb else 0.0,     # wickets per ball
            "rr": (trr / tb) if tb else 0.0,    # running runs conceded per ball
            "br": (tbr / tb) if tb else 0.0,    # boundary runs conceded per ball
        }

    def rate_vol(d, ph, style):
        b, lg = d["balls"], league[ph]
        if style == "attack":
            return (d["wkts"] + K * lg["wr"]) / (b + K), float(d["wkts"])
        if style == "contain":
            conceded = (d["rr"] + K * lg["rr"]) / (b + K)
            return (1.0 / conceded if conceded > 0 else 0.0), float(b)
        # defend
        conceded = (d["br"] + K * lg["br"]) / (b + K)
        return (1.0 / conceded if conceded > 0 else 0.0), float(b)

    qual = [n for n, p in players.items() if p["bowling"]["legal_balls"] >= FIT_QUALIFY_BALLS]

    fit = {n: {ph: {} for ph in PHASES} for n in players}
    for ph in PHASES:
        for s in BOWL_STYLES:
            rates, vols = {}, {}
            for n, p in players.items():
                r, v = rate_vol(p["bowl_phase"][ph], ph, s)
                rates[n], vols[n] = r, v
            ref_names = [n for n in qual if players[n]["bowl_phase"][ph]["balls"] > 0]
            E = _exp_from_spread([vols[n] for n in ref_names], [rates[n] for n in ref_names])
            scores = {n: (vols[n] ** E) * rates[n] if vols[n] > 0 else 0.0 for n in players}
            grid = _percentiles(scores, ref_names)
            for n in players:
                fit[n][ph][s] = grid[n]

    for name, p in players.items():
        p["bowl_fit"] = fit[name]

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

# Hand-curated: no bowling-style field exists anywhere in raw Cricsheet data, so
# this list IS the source of truth for pace/spin.
#
# It is load-bearing in three places -- the model's `is_spin` input, the batter's
# vs-spin/vs-pace matchup features, and the pitch layer's dusty/green matchup --
# so a missing name is not cosmetic. Under the original 55-name list Rashid Khan,
# Narine, Axar Patel, Chawla and Krunal Pandya were all typed as PACE, which
# meant a dusty pitch handed Rashid the pace PENALTY instead of the spin bonus.
#
# The additions below were found by cross-checking the list against the data:
# a STUMPING only happens with the keeper standing up, which in practice means
# spin, so any bowler with stumpings off their bowling is a spinner. That caught
# the 21 high-confidence cases; single-stumping and zero-stumping names were then
# resolved by hand (a genuine spinner who never induced a stumping is invisible
# to the heuristic, and a pace bowler can pick up a freak one -- Bhuvneshwar
# Kumar has 2 in 4503 balls and is emphatically not a spinner).
#
# Anyone still unlisted defaults to "Pace", which remains the safer default.
KNOWN_SPINNERS = {
    # --- original list -----------------------------------------------------
    "A Kumble", "A Mishra", "A Zampa", "AU Rashid", "AM Ghazanfar", "DL Vettori", "GB Hogg", "IS Sodhi",
    "Imran Tahir", "J Suchith", "Jalaj S Saxena", "KA Maharaj", "Kuldeep Yadav", "M Kartik", "M Markande",
    "M Muralitharan", "M Theekshana", "Mohammad Hafeez", "Mujeeb Ur Rahman", "Noor Ahmad", "Parvez Rasool",
    "PP Ojha", "PV Tambe", "R Sai Kishore", "Ravi Bishnoi", "S Badree", "S Gopal", "S Lamichhane", "S Nadeem",
    "S Randiv", "SB Jakati", "SB Joshi", "Shahid Afridi", "Shoaib Malik", "SK Warne", "SMSM Senanayake",
    "Swapnil Singh", "T Shamsi", "YS Chahal", "Zeeshan Ansari", "CV Varun", "M Siddharth", "Harsh Dubey",
    "Suyash Sharma", "HR Shokeen", "K Kartikeya", "KP Appanna", "Mayank Dagar", "MB Parmar",
    "R Ashwin", "RA Jadeja", "Washington Sundar", "Harbhajan Singh", "YK Pathan",
    "RD Chahar",

    # --- front-line spinners the list missed entirely -----------------------
    # All confirmed by stumpings off their own bowling.
    "Rashid Khan", "SP Narine", "AR Patel", "PP Chawla", "KH Pandya", "Shakib Al Hasan",
    "KV Sharma", "R Tewatia", "P Negi", "PWH de Silva", "Iqbal Abdulla", "J Botha",
    "MJ Santner", "M Ashwin", "A Chandila", "Karanveer Singh", "R Sharma",
    "MM Ali", "Shahbaz Ahmed", "Harmeet Singh", "Bipul Sharma", "BAW Mendis",
    "K Gowtham", "Harpreet Brar", "RR Powar", "RE van der Merwe", "AG Murtaza",
    "J Yadav", "Ankit Sharma", "Lalit Yadav", "Mohammad Nabi",
    "KC Cariappa", "WG Jacks", "S Ladda",

    # --- part-time spinners --------------------------------------------------
    # They bowl few overs, but when they do it is spin, and the matchup features
    # need the type right rather than the workload.
    "Yuvraj Singh", "SK Raina", "GJ Maxwell", "CH Gayle", "A Symonds", "JP Duminy",
    "ST Jayasuriya", "TM Dilshan", "DJ Hussey", "AK Markram", "LS Livingstone",
    "N Rana", "Abhishek Sharma", "RG Sharma", "DJ Hooda", "R Parag", "MN Samuels",
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
                            "bowling": {"runs_conceded": 0, "legal_balls": 0, "wickets": 0},
                            "bat_phase": {ph: _blank_phase() for ph in PHASES},
                            "bowl_phase": {ph: _blank_bowl_phase() for ph in PHASES},
                        }
            
            for inning in match.get("innings", []):
                for over_data in inning.get("overs", []):
                    phase = _phase_of(over_data.get("over", 0))
                    for delivery in over_data.get("deliveries", []):
                        batter = delivery.get("batter")
                        bowler = delivery.get("bowler")

                        for p in [batter, bowler]:
                            if p and p not in players:
                                players[p] = {
                                    "name": p,
                                    "batting": {"runs": 0, "balls": 0, "fours": 0, "sixes": 0, "dismissals": 0},
                                    "bowling": {"runs_conceded": 0, "legal_balls": 0, "wickets": 0},
                                    "bat_phase": {ph: _blank_phase() for ph in PHASES},
                                    "bowl_phase": {ph: _blank_bowl_phase() for ph in PHASES},
                                }
                                
                        runs_data = delivery.get("runs", {})
                        batter_runs = runs_data.get("batter", 0)
                        
                        extras_data = delivery.get("extras", {})
                        wides = extras_data.get("wides", 0)
                        noballs = extras_data.get("noballs", 0)
                        
                        bph = players[batter]["bat_phase"][phase]
                        if wides == 0:
                            players[batter]["batting"]["balls"] += 1
                            # per-phase ball detail (only legal balls faced count
                            # toward the style-fit rates, same as career balls)
                            bph["balls"] += 1
                            bph["runs"] += batter_runs
                            if batter_runs == 0:
                                bph["dots"] += 1
                            elif batter_runs == 1:
                                bph["ones"] += 1
                            elif batter_runs == 2:
                                bph["twos"] += 1
                            elif batter_runs == 3:
                                bph["threes"] += 1
                            elif batter_runs == 4:
                                bph["fours"] += 1
                            elif batter_runs == 6:
                                bph["sixes"] += 1

                        players[batter]["batting"]["runs"] += batter_runs
                        if batter_runs == 4:
                            players[batter]["batting"]["fours"] += 1
                        elif batter_runs == 6:
                            players[batter]["batting"]["sixes"] += 1
                            
                        bowler_runs_conceded = batter_runs + wides + noballs
                        players[bowler]["bowling"]["runs_conceded"] += bowler_runs_conceded
                        
                        if wides == 0 and noballs == 0:
                            players[bowler]["bowling"]["legal_balls"] += 1
                            # per-phase bowling ledger for the bowl-fit grid
                            bwph = players[bowler]["bowl_phase"][phase]
                            bwph["balls"] += 1
                            if batter_runs in (4, 6):
                                bwph["br"] += batter_runs
                            elif batter_runs in (1, 2, 3):
                                bwph["rr"] += batter_runs

                        for wicket in delivery.get("wickets", []):
                            player_out = wicket.get("player_out")
                            dismissal_kind = wicket.get("kind", "")
                            
                            if player_out == batter:
                                players[batter]["batting"]["dismissals"] += 1
                                bph["dismissals"] += 1
                            elif player_out in players:
                                players[player_out]["batting"]["dismissals"] += 1
                                # runouts etc. can dismiss the non-striker; credit
                                # it to the phase this ball was bowled in
                                players[player_out]["bat_phase"][phase]["dismissals"] += 1
                                
                            if dismissal_kind in BOWLER_CREDITED_DISMISSALS:
                                players[bowler]["bowling"]["wickets"] += 1
                                players[bowler]["bowl_phase"][phase]["wkts"] += 1
                                
                            if dismissal_kind == "stumped":
                                for fielder in wicket.get("fielders", []):
                                    dynamic_keepers.add(fielder.get("name"))
                                
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
            
        processed_count += 1
        if processed_count % 200 == 0:
            print(f"Processed {processed_count}/{total_files} matches...")

    # Phase 1.5: derive per-player style-fit tables from the phase splits
    compute_style_fits(players)
    compute_bowler_fits(players)

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
            "bowling_style": "Spin" if name in KNOWN_SPINNERS else "Pace",
            "role": data["role"],
            "roles": data["roles"],
            "signature": data["signature"],
            "style_fit": data["style_fit"],
            "bowl_fit": data["bowl_fit"],
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
