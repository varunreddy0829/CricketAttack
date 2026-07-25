"""
Cricket Attack - Authoritative multiplayer game server.

This server owns ALL shared game state. Two devices connect to a single game
(one creates, one joins with a code) and poll GET /api/state for a *redacted*
view of the world. The batting side never receives the bowling side's intent
and vice-versa; intent is only revealed after both sides submit an over and the
math engine resolves it.

The math engine (src/engine/*, src/models/*) is reused unchanged. This module
re-implements the over loop (simulate_over_rich) so it can capture per-ball,
per-batter and per-bowler detail the UI needs, while still calling the engine's
calculate_single_ball for the actual probability math.
"""

import os
import sys
import json
import time
import random
import string
import secrets
import threading

from flask import Flask, jsonify, request, send_from_directory

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.player import Batter, Bowler
from src.models.match_state import MatchState
from src.engine.simulator import calculate_single_ball, EXTRAS_PROB
from src.engine.conditions import phase_for_over
from src.engine.draft_generator import generate_draft_pool, auction_pool_size_per_set

app = Flask(__name__, static_folder="public")

@app.after_request
def add_header(response):
    response.cache_control.no_store = True
    response.cache_control.no_cache = True
    response.cache_control.max_age = 0
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    return response

# Absolute, based on this file's own location (not the process's working
# directory) -- WSGI hosts (PythonAnywhere, gunicorn, etc.) don't guarantee
# the CWD is the repo root when they import this module.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORICAL_PATH = os.path.join(REPO_ROOT, "data", "players_historical.json")

OVERS_PER_INNINGS = 20
MAX_OVERS_PER_BOWLER = 4

# --- Auction rules (mirrors the original main.js hotseat logic) ---------------
INITIAL_PURSE = 100.0        # Cr
SQUAD_MIN, SQUAD_MAX = 15, 21
XI_SIZE = 11
XI_MAX_OVERSEAS = 5
BASE_BID = 0.5               # minimum bid increment
# Opening (base) price per tier, in Cr. Group 3 = 0.2 Cr (20 lakh).
TIER_OPENING = {"Marquee": 2.0, "Mid-Level": 1.0, "Group 3": 0.2}
TIER_WAIT = {"Marquee": 5.0, "Mid-Level": 3.5, "Group 3": 2.0}  # seconds per strike
ANNOUNCE_WAIT = 2.5          # pause on SOLD / UNSOLD before the next lot

# --- Load the authoritative player database once -----------------------------

def _load_historical():
    with open(HISTORICAL_PATH, "r", encoding="utf-8") as f:
        text = f.read()
        return json.loads(text[text.find("["):])

ALL_PLAYERS = _load_historical()
BY_NAME = {p["name"]: p for p in ALL_PLAYERS}

# Stage-2 (strike rate vs economy) and Stage-3 (wicket factor) benchmarks are
# tunable from config/baseline_weights.json -- see that file for the formula
# shapes and src/engine/stats_calculator.py for how they're consumed. Every
# score follows VOLUME^A x RATE^B, ghost-stat (Bayesian pseudo-count) smoothed
# toward a league prior so small samples (e.g. a keeper who bowled one ball
# and took a wicket, or a batter with 449 freakish balls) regress toward
# average instead of dominating. *_base values are league medians of the raw
# scores over settled regulars (>=regular_balls_cutoff balls), computed here
# from the live data, so an average regular always scores 1.0 -- editing the
# config and restarting re-tunes the engine without touching code.
BASELINE_WEIGHTS_PATH = os.path.join(REPO_ROOT, "config", "baseline_weights.json")

def _median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0

def _load_baseline_weights():
    with open(BASELINE_WEIGHTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

BASELINE_WEIGHTS = _load_baseline_weights()

# Stadiums + pitch personalities (compiled from the raw match data's venue
# fields — see config/ground_configs.json). Teams claim unique home grounds
# after the auction; the pitch feeds the engine's conditions stage.
GROUND_CONFIGS_PATH = os.path.join(REPO_ROOT, "config", "ground_configs.json")

def _load_grounds():
    with open(GROUND_CONFIGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

_GROUNDS_CFG = _load_grounds()
STADIUMS = _GROUNDS_CFG["stadiums"]
STADIUM_BY_ID = {s["id"]: s for s in STADIUMS}
PITCH_DESCRIPTIONS = _GROUNDS_CFG["pitch_types"]

def _compute_league_avg():
    cfg = BASELINE_WEIGHTS
    s2 = cfg["stage2_strike_rate_economy"]
    s3 = cfg["stage3_wicket_factor"]
    s2_bat, s2_bowl = s2["batting"], s2["bowling"]
    s3_bowl, s3_bat = s3["bowling"], s3["batting"]
    regular_balls = cfg["regular_balls_cutoff"]

    # Stage 3: bowler threat = wickets^A / ghost_sr^B; batter patience = runs^A * ghost_avg^B
    def ghost_sr_s3(p):
        b = p["bowling"]
        return (b["legal_balls"] + s3_bowl["K_wickets"] * s3_bowl["prior_sr"]) / (b["wickets"] + s3_bowl["K_wickets"])
    def ghost_avg_s3(p):
        b = p["batting"]
        return (b["runs"] + s3_bat["K_dismissals"] * s3_bat["prior_avg"]) / (b["dismissals"] + s3_bat["K_dismissals"])

    # Stage 2: batting power = balls^A * ghost_sr^B; bowling power = legal_balls^A / ghost_eco^B
    def ghost_sr_s2(p):
        b = p["batting"]
        return ((b["runs"] + (s2_bat["prior_sr"] / 100.0) * s2_bat["K_balls"]) / (b["balls"] + s2_bat["K_balls"])) * 100.0
    def ghost_eco_s2(p):
        b = p["bowling"]
        return ((b["runs_conceded"] + s2_bowl["prior_eco"] * s2_bowl["K_balls"]) / (b["legal_balls"] + s2_bowl["K_balls"])) * 6.0

    threat_scores = [
        (p["bowling"]["wickets"] ** s3_bowl["A_volume"]) / (ghost_sr_s3(p) ** s3_bowl["B_rate"])
        for p in ALL_PLAYERS
        if p["bowling"]["legal_balls"] >= regular_balls and p["bowling"]["wickets"] > 0
    ]
    patience_scores = [
        (p["batting"]["runs"] ** s3_bat["A_volume"]) * (ghost_avg_s3(p) ** s3_bat["B_rate"])
        for p in ALL_PLAYERS
        if p["batting"]["balls"] >= regular_balls and p["batting"]["dismissals"] > 0
    ]
    bat_power_scores = [
        (p["batting"]["balls"] ** s2_bat["A_volume"]) * (ghost_sr_s2(p) ** s2_bat["B_rate"])
        for p in ALL_PLAYERS
        if p["batting"]["balls"] >= regular_balls
    ]
    bowl_power_scores = [
        (p["bowling"]["legal_balls"] ** s2_bowl["A_volume"]) / (ghost_eco_s2(p) ** s2_bowl["B_rate"])
        for p in ALL_PLAYERS
        if p["bowling"]["legal_balls"] >= regular_balls and ghost_eco_s2(p) > 0
    ]
    return {
        # Stage 3
        "prior_sr": s3_bowl["prior_sr"],
        "prior_avg": s3_bat["prior_avg"],
        "k_wickets": s3_bowl["K_wickets"],
        "k_dismissals": s3_bat["K_dismissals"],
        "threat_a": s3_bowl["A_volume"],
        "threat_b": s3_bowl["B_rate"],
        "patience_a": s3_bat["A_volume"],
        "patience_b": s3_bat["B_rate"],
        "threat_base": _median(threat_scores) if threat_scores else 0.302,
        "patience_base": _median(patience_scores) if patience_scores else 315.0,
        "patience_floor": s3_bat["patience_floor"],
        "wicket_damp": s3["damp"],
        # Stage 2
        "prior_bat_sr": s2_bat["prior_sr"],
        "prior_eco": s2_bowl["prior_eco"],
        "k_balls_sr": s2_bat["K_balls"],
        "k_balls_eco": s2_bowl["K_balls"],
        "bat_power_a": s2_bat["A_volume"],
        "bat_power_b": s2_bat["B_rate"],
        "bowl_power_a": s2_bowl["A_volume"],
        "bowl_power_b": s2_bowl["B_rate"],
        "bat_power_base": _median(bat_power_scores) if bat_power_scores else 1316.7,
        "bowl_power_base": _median(bowl_power_scores) if bowl_power_scores else 0.846,
        "bat_power_floor": s2_bat["power_floor"],
        "bowl_power_floor": s2_bowl["power_floor"],
        "str_damp": s2["damp"],
        # Stage 4
        "intent_strength": cfg["stage4_intent"]["strength"],
    }

LEAGUE_AVG = _compute_league_avg()

# --- Multi-game registry -----------------------------------------------------
# GAMES holds every live game, keyed by its 4-char join code, so unrelated
# friend groups can play concurrent, fully isolated matches on one server.
# TOKEN_TO_CODE maps a player's session token to which game they belong to.
#
# GAME itself stays a bare module-level name (unchanged from the original
# single-game version) but is now *resolved per request*: every route looks up
# the right game by token/code and rebinds GAME to it before touching any game
# state, all while holding LOCK -- so the huge amount of existing code below
# that references GAME directly (as a bare name, not a parameter) keeps
# working unchanged. Because every request serializes on the same LOCK for its
# whole duration, rebinding GAME this way is race-free: no other thread can
# observe or mutate GAME while the current request holds the lock. This is a
# simple single-lock-for-all-games design, not per-game locking -- at this
# project's scale (lightweight polling, no heavy per-request computation)
# that's not a real bottleneck, and it avoids a much more complex, bug-prone
# fine-grained locking scheme for no practical benefit.
GAMES = {}
TOKEN_TO_CODE = {}
GAME = None
LOCK = threading.RLock()

# A game with no activity (no mutation and no /api/state poll) for this long
# is assumed abandoned and is swept from GAMES to bound memory growth --
# players who leave mid-lobby without explicitly forfeiting are the common
# case this catches (an explicit exit/forfeit already ends the game sooner).
GAME_TTL_SECONDS = 4 * 3600
_last_sweep_at = 0.0


def _new_code():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=4))


def _register_game(g):
    """Store a freshly created game, guarding against the astronomically
    unlikely event that its random code collides with another live game."""
    while g["code"] in GAMES:
        g["code"] = _new_code()
    GAMES[g["code"]] = g
    return g


def _game_by_code(code):
    return GAMES.get((code or "").strip().upper())


def _game_by_token(token):
    code = TOKEN_TO_CODE.get(token)
    return GAMES.get(code) if code else None


def _register_token(code, token):
    TOKEN_TO_CODE[token] = code


def _sweep_stale_games():
    """Drop games untouched for GAME_TTL_SECONDS. Throttled to once a minute
    (called from the 0.2s auction tick, which already holds LOCK) so it
    doesn't walk the whole registry on every tick."""
    global _last_sweep_at
    now = time.time()
    if now - _last_sweep_at < 60.0:
        return
    _last_sweep_at = now
    stale_codes = [code for code, g in GAMES.items()
                   if now - g.get("last_seen", now) > GAME_TTL_SECONDS]
    for code in stale_codes:
        del GAMES[code]
    if stale_codes:
        dead_tokens = [tok for tok, code in TOKEN_TO_CODE.items() if code in stale_codes]
        for tok in dead_tokens:
            del TOKEN_TO_CODE[tok]


TEAM_COLOR_PALETTE = ["#e6483c", "#3b82c4", "#e0b400", "#2f9e5c",
                      "#9d5ce0", "#e07b2e", "#22a6a6", "#e05c9e"]


def _fresh_game(num_teams=2):
    """A plain 1v1 game is just num_teams=2 — the generalized N-team registry
    (team_ids/teams keyed "team1".."teamN") transparently degrades to the
    original 2-key shape, so the existing 1v1 flow is unaffected. Tournament
    mode (num_teams 3-8) reuses the exact same structure."""
    team_ids = [f"team{i+1}" for i in range(num_teams)]
    return {
        "code": _new_code(),
        "version": 1,
        "last_seen": time.time(),
        "phase": "lobby",  # lobby -> [auction -> xi ->]* match -> finished
        "tokens": {},       # token -> team_id
        "team_ids": team_ids,          # ordered list of every team slot in this game
        "teams": {t: {"name": f"Team {i+1}", "joined": False, "xi": [], "ready": False,
                      "color": TEAM_COLOR_PALETTE[i % len(TEAM_COLOR_PALETTE)]}
                  for i, t in enumerate(team_ids)},
        "match_teams": team_ids[:2],   # the two teams contesting the CURRENT match
        # match state (populated when the match starts)
        "innings": 1,
        "batting_side": team_ids[0],
        "target": None,
        "state": None,            # MatchState
        "bowler": None,           # active Bowler object
        "bowler_stats": {},       # name -> overs bowled this innings
        "last_bowler": None,
        "used_batters": [],       # names that have come to the crease this innings
        "bat_card": {},           # name -> batting scorecard row
        "bowl_card": {},          # name -> bowling scorecard row
        "completed_innings": [],  # snapshots for the scorecard tab
        "motm": None,             # this match's Man of the Match (set once it finishes)
        "this_over": [],          # structured commentary for the last completed over
        "live": [],               # full running commentary (all overs, both innings)
        "pending_over": {
            "batting": {"submitted": False, "striker_intent": 50, "non_striker_intent": 50},
            "bowling": {"submitted": False, "bowler_name": None, "bowl_intent": 50},
        },
        "active_over": None,      # locked intents once both sides submit
        "over_end_at": 0,
        "over_start_runs": 0,     # for the end-of-over summary
        "over_start_wickets": 0,
        "result": None,
        "match_winner": None,     # team_id, or None for a tie (see _compute_result)
        "abandoned": False,          # someone exited -> opponent wins
        "abandoned_by": None,
        # match flow state machine (within phase == "match"):
        #   toss -> openers -> play -> [await_batter -> await_resume] / free_hit
        "stage": "lobby",
        "toss": {"winner": None, "decided": False, "choice": None},
        "next_ball_free_hit": False,
        "free_hit": {"active": False, "batting_ready": False, "bowling_ready": False,
                     "striker_intent": 50, "non_striker_intent": 50, "bowl_intent": 50,
                     "striker_role": DEFAULT_BAT_ROLE, "non_striker_role": DEFAULT_BAT_ROLE,
                     "bowl_role": DEFAULT_BOWL_ROLE},
        # auction / squad / XI selection (phase == "auction" / "grounds" / "xi")
        "start_votes": {t: False for t in team_ids},   # every team must agree to start the auction
        "auction": None,
        "squads": None,
        "xi_select": None,
        "ground_pick": None,      # phase == "grounds": {team: {"ground": id|None, "locked": bool}}
        "match_ground": None,     # stadium dict for the CURRENT match (set at match start)
        "gambit_cards": None,     # {team: {"attack": bool_available, "trap": bool_available}} per match
        "tournament": None,   # populated only for tournament games (see _start_tournament)
    }


def _bump():
    GAME["version"] += 1
    GAME["last_seen"] = time.time()


# --- Roster helpers ----------------------------------------------------------

def _energy_mult(name):
    """Tournament fatigue: each consecutive match a player appears in costs
    0.5% of his effective OVR, restored fully by sitting one game out. Small
    per match, meaningful over a long unbroken run (see _finish_tournament_fixture
    for the bookkeeping). Non-tournament games are unaffected."""
    t = GAME.get("tournament") if GAME else None
    if not t:
        return 1.0
    fatigue = (t.get("fatigue") or {}).get(name, 0)
    return max(0.95, 1.0 - 0.005 * fatigue)


def _make_batter(record):
    b = record["batting"]
    return Batter(
        name=record["name"],
        ovr=max(1, round(record["batting_ovr"] * _energy_mult(record["name"]))),
        career_runs=b["runs"],
        career_balls=b["balls"],
        fours=b["fours"],
        sixes=b["sixes"],
        dismissals=max(1, b["dismissals"]),
        intent=50,
    )


def _make_bowler(record, intent=50):
    bw = record["bowling"]
    return Bowler(
        name=record["name"],
        ovr=max(1, round(record["bowling_ovr"] * _energy_mult(record["name"]))),
        eco=bw["eco"] if bw["eco"] and bw["eco"] > 0 else 8.5,
        wkt=bw["wickets"],
        intent=intent,
        legal_balls=bw["legal_balls"],
        style=record.get("bowling_style", "Pace"),
    )


# --- Roles (Stage 4) -------------------------------------------------------
# The batting/bowling role replaces the old intent slider. Each ball, the
# batter's role and the bowler's role are turned into a 0-99 grid lookup for
# the current phase, then applied by the engine's apply_roles (see roles.py).
BAT_ROLES = ("attack", "rotate", "defend")
BOWL_ROLES = ("attack", "contain", "defend")
DEFAULT_BAT_ROLE = "rotate"
DEFAULT_BOWL_ROLE = "contain"
# batting role -> which style_fit cell holds its grid (defend reads the anchor cell)
_BAT_ROLE_CELL = {"attack": "attack", "rotate": "rotate", "defend": "anchor"}
# role -> a pseudo-intent so the Trap gambit (which reads aggression) still works
_ROLE_INTENT = {"attack": 70, "rotate": 50, "defend": 35}


def _phase_key(over_num):
    if over_num <= 5:
        return "pp"
    if over_num <= 14:
        return "mid"
    return "death"


def _clean_bat_role(v):
    return v if v in BAT_ROLES else DEFAULT_BAT_ROLE


def _clean_bowl_role(v):
    return v if v in BOWL_ROLES else DEFAULT_BOWL_ROLE


def _bat_grid(name, over_num, role):
    """The batter's 0-99 grid for this phase + role (50 if unknown)."""
    rec = BY_NAME.get(name) or {}
    sf = rec.get("style_fit") or {}
    cell = _BAT_ROLE_CELL.get(role, "rotate")
    return (sf.get(_phase_key(over_num)) or {}).get(cell, 50)


def _bowl_grid(name, over_num, role):
    """The bowler's 0-99 grid for this phase + role (50 if unknown)."""
    rec = BY_NAME.get(name) or {}
    bf = rec.get("bowl_fit") or {}
    return (bf.get(_phase_key(over_num)) or {}).get(role, 50)


def _card_fields(record):
    """Public, UI-facing fields for a player card."""
    return {
        "name": record["name"],
        "batting_ovr": record["batting_ovr"],
        "bowling_ovr": record["bowling_ovr"],
        "is_foreigner": record.get("is_foreigner", False),
        "is_keeper": record.get("is_keeper", False),
        "bowling_style": record.get("bowling_style", "Pace"),
        "role": record.get("role"),
        "roles": record.get("roles"),
        "signature": record.get("signature"),
        "style_fit": record.get("style_fit"),
        "bowl_fit": record.get("bowl_fit"),
    }


def _role_meta(name):
    """The card-flip fields (roles, signature, 3x3 style_fit + bowl_fit) for a
    player by name -- spread into card dicts built by hand, not via _card_fields."""
    r = BY_NAME.get(name, {})
    return {"role": r.get("role"), "roles": r.get("roles"),
            "signature": r.get("signature"), "style_fit": r.get("style_fit"),
            "bowl_fit": r.get("bowl_fit")}


def _auto_two_xis():
    """Draft two balanced, realistic squads for Quick Match: an 11-player XI
    (6 top specialist batsmen at the top of the order + 5 frontline bowlers in
    the tail, per team) plus 2 spare reserves each (1 more batter, 1 more
    bowler) so Impact Player has an actual bench to draw from -- Quick Match
    skips the draft UI, but that shouldn't mean skipping the feature entirely.
    Batting order (top batsmen first) makes the lower order behave like a real
    tail rather than 6 pure bowlers collapsing.

    Returns (xi1, xi2, roster1, roster2) -- roster includes the XI plus the
    2 reserves, in the shape GAME["squads"][t]["roster"] expects."""
    bats = sorted([p for p in ALL_PLAYERS if p["batting"]["balls"] >= 300],
                  key=lambda p: -p["batting_ovr"])[:40]
    bowls = sorted([p for p in ALL_PLAYERS if p["bowling"]["legal_balls"] >= 300],
                   key=lambda p: -p["bowling_ovr"])[:40]
    random.shuffle(bats)
    random.shuffle(bowls)

    used = set()
    def take(src, n, team):
        got = 0
        for p in src:
            if got >= n:
                break
            if p["name"] in used:
                continue
            used.add(p["name"])
            team.append(p)
            got += 1

    t1, t2 = [], []
    take(bats, 6, t1)
    take(bats, 6, t2)
    take(bowls, 5, t1)
    take(bowls, 5, t2)

    # safety fill from the full DB if the filtered pools ran short
    filler = (p for p in ALL_PLAYERS if p["name"] not in used)
    for team in (t1, t2):
        while len(team) < 11:
            p = next(filler)
            used.add(p["name"])
            team.append(p)

    reserves = {"team1": [], "team2": []}
    for key, team in (("team1", t1), ("team2", t2)):
        take(bats, 1, reserves[key])
        take(bowls, 1, reserves[key])
        while len(reserves[key]) < 2:
            p = next(filler)
            used.add(p["name"])
            reserves[key].append(p)

    xi1, xi2 = [_card_fields(p) for p in t1[:11]], [_card_fields(p) for p in t2[:11]]
    roster1 = xi1 + [_card_fields(p) for p in reserves["team1"]]
    roster2 = xi2 + [_card_fields(p) for p in reserves["team2"]]
    return xi1, xi2, roster1, roster2


# --- Match lifecycle ---------------------------------------------------------

def _prepare_innings(batting_side, target=None, keep_this_over=False):
    """Set up a fresh innings but leave the crease empty — the batting side picks
    its openers (stage 'openers') before any ball is bowled.

    keep_this_over=True is used for the innings-1 -> innings-2 handoff: the
    final over of innings 1 was just simulated in this very request, and
    wiping this_over here (before the client's next /api/state poll can ever
    see it) silently drops its commentary. Leaving it be for now is safe --
    innings 2's own first _try_resolve_over() call clears this_over itself
    once real balls start, and the full-screen openers overlay covers the
    match view in the meantime anyway."""
    g = GAME
    g["batting_side"] = batting_side
    g["target"] = target
    lineup = [_make_batter(BY_NAME[p["name"]]) for p in g["teams"][batting_side]["xi"]]
    state = MatchState(lineup)
    state.target = target
    state.striker_index = None
    state.non_striker_index = None
    g["state"] = state
    g["bowler"] = None
    g["bowler_stats"] = {}
    g["last_bowler"] = None
    g["bat_card"] = {}
    g["bowl_card"] = {}
    if not keep_this_over:
        g["this_over"] = []
    g["active_over"] = None
    g["over_end_at"] = 0
    g["used_batters"] = []
    g["vacant_slot"] = "striker"   # which crease slot set_next_batter should fill
    g["next_ball_free_hit"] = False
    g["free_hit"] = {"active": False, "batting_ready": False, "bowling_ready": False,
                     "striker_intent": 50, "non_striker_intent": 50, "bowl_intent": 50,
                     "striker_role": DEFAULT_BAT_ROLE, "non_striker_role": DEFAULT_BAT_ROLE,
                     "bowl_role": DEFAULT_BOWL_ROLE}
    _reset_pending()
    g["stage"] = "openers"


def _do_toss():
    """Coin toss at the start of the match; winner chooses to bat or bowl.
    Always between the two teams in GAME['match_teams'] — for a plain 1v1 game
    that's the whole game; for a tournament fixture it's whichever two teams
    the current fixture pairs up."""
    GAME["toss"] = {"winner": random.choice(GAME["match_teams"]), "decided": False, "choice": None}
    GAME["stage"] = "toss"


def _reset_pending():
    GAME["pending_over"] = {
        "batting": {"submitted": False, "striker_intent": 50, "non_striker_intent": 50,
                    "striker_role": DEFAULT_BAT_ROLE, "non_striker_role": DEFAULT_BAT_ROLE},
        "bowling": {"submitted": False, "bowler_name": None, "bowl_intent": 50,
                    "bowl_role": DEFAULT_BOWL_ROLE},
    }
    for t in GAME["match_teams"]:
        GAME["teams"][t]["ready"] = False


def _bowling_side():
    """The other of the two teams in the CURRENT match (GAME['match_teams']),
    not 'the other of all teams in the game' — the distinction only matters in
    tournament mode, where more than 2 teams may exist overall."""
    a, b = GAME["match_teams"]
    return b if GAME["batting_side"] == a else a


def _ensure_bat_row(name):
    if name not in GAME["bat_card"]:
        rec = BY_NAME[name]
        GAME["bat_card"][name] = {
            "name": name,
            "runs": 0, "balls": 0, "fours": 0, "sixes": 0,
            "out": False, "how_out": "not out",
            "batting_ovr": rec["batting_ovr"],
        }


def _ensure_bowl_row(name):
    if name not in GAME["bowl_card"]:
        rec = BY_NAME[name]
        GAME["bowl_card"][name] = {
            "name": name,
            "balls": 0, "runs": 0, "wickets": 0,
            "bowling_ovr": rec["bowling_ovr"],
        }


def _overs_str(balls):
    return f"{balls // 6}.{balls % 6}"


# --- Commentary flavour -------------------------------------------------------
# Ball-by-ball lines with variety. {b} = batter, {bl} = bowler. Purely cosmetic;
# the outcome is already decided by the engine before a line is picked.
COMMENTARY = {
    "dot": [
        "{bl} nails the length — {b} can't get it away.",
        "Beaten outside off! Dot ball.",
        "{b} defends watchfully. No run.",
        "Tight from {bl}, nothing on offer.",
        "Big appeal... turned down! No run.",
        "{b} pushes straight to the fielder.",
        "Swing and a miss! {bl} grins.",
        "Right in the block-hole, dug out by {b}.",
    ],
    "one": [
        "Worked into the gap — easy single for {b}.",
        "Quick single! Sharp running.",
        "{b} nudges it to the leg side for one.",
        "Dropped and run, one to {b}.",
        "Pushed to the sweeper, a comfortable single.",
    ],
    "two": [
        "Driven into the gap, {b} comes back for two.",
        "Good running between the wickets — two runs.",
        "{b} works it into the deep for a couple.",
        "Placed nicely, they scamper two.",
    ],
    "three": [
        "Into the deep! Excellent running, three to {b}.",
        "Three! The fielder does well to keep it in.",
    ],
    "four": [
        "FOUR! {b} threads the covers — glorious shot!",
        "Cracked away! {bl} is punished for FOUR.",
        "FOUR! Beautifully timed down the ground.",
        "Short and wide — {b} slaps it away for FOUR!",
        "FOUR! Flicked off the pads, all along the carpet.",
        "Edged... and FOUR! Lucky runs for {b}.",
    ],
    "five": [
        "FIVE! Overthrows gift {b} a bonus run.",
    ],
    "six": [
        "SIX! {b} launches {bl} into the stands!",
        "MAXIMUM! That has gone miles!",
        "SIX! {b} clears the ropes with ease.",
        "Into the crowd! {bl} has no answer. SIX!",
        "BANG! {b} goes downtown for a huge SIX!",
    ],
    "wicket": [
        "OUT! {bl} castles {b} — timber!",
        "GOT HIM! {b} has to walk, {bl} strikes!",
        "WICKET! {b} nicks off and {bl} roars!",
        "Bowled 'im! {bl} sneaks through {b}'s defence.",
        "GONE! Massive wicket — {b} departs.",
        "Trapped! {bl} pins {b} in front. That's out!",
    ],
    "wide": [
        "Wide! {bl} drifts down the leg side. One extra.",
        "Called wide — a gift of a run.",
    ],
    "noball": [
        "No ball! {bl} oversteps. Free run.",
        "Overstepped! No ball called on {bl}.",
    ],
    "free_dot": [
        "FREE HIT! {b} swings hard and misses — no run!",
        "FREE HIT! Huge heave... and thin air. Nothing off it.",
        "FREE HIT! {b} goes for the stands but misses completely.",
        "FREE HIT! Beaten by pace — the free hit goes begging.",
        "FREE HIT! Wild swing from {b}, no contact. No run.",
    ],
}
_RUNS_KIND = {0: "dot", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

def _say(kind, b="", bl=""):
    return random.choice(COMMENTARY[kind]).format(b=b, bl=bl)


# --- The over simulation (reuses engine math, adds full bookkeeping) ---------

def _simulate_until_pause():
    """Bowl deliveries for the current over until it completes, a wicket falls,
    or the innings ends. Appends structured events to GAME['this_over'] and the
    running GAME['live']. Returns one of:
    'over_complete' | 'wicket_pending' | 'innings_over'.
    """
    g = GAME
    state = g["state"]
    bowler = g["bowler"]
    ao = g["active_over"]
    by_name = ao.setdefault("intent_by_name", {})
    by_role = ao.setdefault("role_by_name", {})

    # Intent follows the PERSON, not the crease slot — looked up by name so a
    # mid-over strike rotation (including one caused by a free-hit ball, or a
    # re-entry after a wicket) can never swap two batters' settings onto each
    # other. Falls back to the slot value for a batter not yet registered by
    # name (a brand-new batter mid-over — see set_next_batter/ready_resume).
    s = state.get_striker()
    ns = state.get_non_striker()
    if s:
        s.intent = by_name.get(s.name, ao["striker_intent"])
    if ns:
        ns.intent = by_name.get(ns.name, ao["non_striker_intent"])
    bowler.intent = ao["bowl_intent"]

    _ensure_bowl_row(bowler.name)

    while state.balls < g["over_end_at"]:
        striker = state.get_striker()
        if striker is None:
            # waiting on a new batter
            return "innings_over" if state.is_all_out() else "wicket_pending"

        ball_no = f"{state.balls // 6}.{(state.balls % 6) + 1}"
        free_ball = g["next_ball_free_hit"]

        # Extras (outside the weighted engine, re-bowled) — skipped on a free hit,
        # which is always a clean legal delivery.
        if not free_ball and random.random() < EXTRAS_PROB:
            is_wide = random.random() < 0.7
            state.add_extra()
            g["bowl_card"][bowler.name]["runs"] += 1
            kind = "wide" if is_wide else "noball"
            _emit(ball_no, "extra", _say(kind, striker.name, bowler.name),
                  outcome=("Wd" if is_wide else "Nb"))
            if g["target"] is not None and state.runs >= g["target"]:
                return "innings_over"
            if not is_wide:
                # No ball -> the next delivery is a FREE HIT; pause for the
                # optional intent window on both sides.
                g["next_ball_free_hit"] = True
                return "free_hit_pending"
            continue

        # A free-hit delivery uses the intents chosen during the free-hit window.
        if free_ball:
            g["next_ball_free_hit"] = False
            fh = g["free_hit"]
            striker.intent = fh["striker_intent"]
            _ns = state.get_non_striker()
            if _ns:
                _ns.intent = fh["non_striker_intent"]
            bowler.intent = fh["bowl_intent"]

        # roles for THIS ball. Roles follow the PERSON (looked up by name), with
        # the free-hit window's choices overriding on a free-hit delivery — same
        # discipline as the intent lookup above so a strike rotation never swaps
        # two batters' roles.
        over_num = state.balls // 6
        if free_ball:
            fh = g["free_hit"]
            bat_role = _clean_bat_role(fh.get("striker_role"))
            bowl_role = _clean_bowl_role(fh.get("bowl_role"))
        else:
            bat_role = _clean_bat_role(by_role.get(striker.name, ao.get("striker_role")))
            bowl_role = _clean_bowl_role(ao.get("bowl_role"))

        # match conditions for THIS ball: home-ground pitch vs bowler style,
        # innings phase, armed gambits, and the batting/bowling roles (Stage 4).
        # striker_intent is derived from the batting role so the Trap gambit
        # (which reads aggression) still responds to Attack vs Defend.
        ctx = {
            "pitch": (g.get("match_ground") or {}).get("pitch"),
            "bowler_style": bowler.style,
            "over_num": over_num,
            "attack_gambit": ao.get("attack_gambit"),
            "trap_gambit": ao.get("trap_gambit"),
            "striker_intent": _ROLE_INTENT.get(bat_role, 50),
            "bat_role": bat_role,
            "bat_grid": _bat_grid(striker.name, over_num, bat_role),
            "bowl_role": bowl_role,
            "bowl_grid": _bowl_grid(bowler.name, over_num, bowl_role),
        }
        outcome = calculate_single_ball(striker, bowler, LEAGUE_AVG, ctx)
        state.add_ball()
        g["bowl_card"][bowler.name]["balls"] += 1
        row = g["bat_card"][striker.name]
        row["balls"] += 1
        fh_prefix = "FREE HIT! " if free_ball else ""

        if outcome == "Out" and free_ball:
            # can't be dismissed on a free hit (bowled/caught) -> treated as a dot
            _emit(ball_no, "run", fh_prefix + "Beaten, but not out on the free hit — no run.", outcome="0")
        elif outcome == "Out":
            row["out"] = True
            row["how_out"] = f"b {bowler.name}"
            bc = g["bowl_card"][bowler.name]
            bc["wickets"] += 1
            _emit(ball_no, "wicket", _say("wicket", striker.name, bowler.name), outcome="W")
            # hat-trick: consecutive wickets on this bowler's own consecutive
            # deliveries (extras don't break the streak -- see the extras
            # branch above, which deliberately leaves last_ball_wicket alone)
            bc["consec_wkts"] = bc.get("consec_wkts", 0) + 1 if bc.get("last_ball_wicket") else 1
            bc["last_ball_wicket"] = True
            if bc["consec_wkts"] == 3:
                _emit(ball_no, "milestone", f"🎩 HAT-TRICK! {bowler.name} takes three wickets in a row!", outcome="HT")
            state.handle_wicket()
            g["vacant_slot"] = "striker"   # the batter facing the ball is always the striker
            if state.is_all_out():
                return "innings_over"
            return "wicket_pending"
        else:
            runs = int(outcome)
            state.add_runs(runs)
            prev_runs = row["runs"]
            row["runs"] += runs
            g["bowl_card"][bowler.name]["runs"] += runs
            g["bowl_card"][bowler.name]["last_ball_wicket"] = False
            if runs == 4:
                row["fours"] += 1
            elif runs == 6:
                row["sixes"] += 1
            # milestones: guarded so a player already past a threshold never
            # re-fires it on a later ball, and the two are mutually exclusive
            # (a single ball can cross at most one, since max runs/ball = 6)
            if prev_runs < 50 <= row["runs"] < 100:
                _emit(ball_no, "milestone", f"🎉 FIFTY! {striker.name} brings up his half-century!", outcome="50")
            elif prev_runs < 100 <= row["runs"]:
                _emit(ball_no, "milestone", f"💯 CENTURY! {striker.name} reaches three figures!", outcome="100")
            label = "boundary" if runs in (4, 6) else "run"
            if free_ball and runs == 0:
                text = _say("free_dot", striker.name, bowler.name)   # swing-and-miss, not a block
            else:
                text = fh_prefix + _say(_RUNS_KIND[runs], striker.name, bowler.name)
            _emit(ball_no, label, text, outcome=str(runs))
            if g["target"] is not None and state.runs >= g["target"]:
                return "innings_over"
            if runs in (1, 3, 5):
                state.rotate_strike()

        # after a free-hit ball, restore the over's locked intents for the
        # rest — BY NAME (a free-hit single can itself rotate strike, and a
        # slot-based restore here would swap the two batters' intents)
        if free_ball:
            _s = state.get_striker()
            _ns2 = state.get_non_striker()
            if _s:
                _s.intent = by_name.get(_s.name, ao["striker_intent"])
            if _ns2:
                _ns2.intent = by_name.get(_ns2.name, ao["non_striker_intent"])
            bowler.intent = ao["bowl_intent"]

    # over complete: rotate strike for the new over
    if state.get_striker() is not None:
        state.rotate_strike()
    return "over_complete"


def _emit(ball, ev_type, text, outcome=None, extra=False):
    entry = {
        "ball": ball, "type": ev_type, "text": text, "outcome": outcome,
        "over": GAME["innings"],
        "bowler": GAME["bowler"].name if GAME["bowler"] else None,
    }
    GAME["this_over"].append(entry)
    GAME["live"].append(entry)


def _apply_impact_sub(role, out_name, in_name):
    """One bench-for-XI swap per team, once for the whole match. Allowed only
    at safe checkpoints (never mid-over): the batting side may swap between
    overs, right after a wicket (filling the open crease slot), or during the
    resume-confirmation window that follows (a fresh batter just walked in --
    still "after a wicket"); the bowling side may swap between overs, or
    during that same resume window (a fresh batter is exactly when a bowling
    side would want to react). Who's in your XI isn't hidden information
    (only intents/bowler-before-lock are), so the swap is revealed to both
    sides immediately via commentary."""
    g = GAME
    if not g.get("squads") or role not in g["squads"]:
        raise ValueError("No bench to bring on for this match.")
    ip = (g.get("impact_player") or {}).get(role)
    if ip is None:
        raise ValueError("Impact Player isn't available in this match.")
    if ip["used"]:
        raise ValueError("You've already used your Impact Player this match.")

    i_bat = role == g["batting_side"]
    stage = g["stage"]
    my_pending = g["pending_over"]["batting" if i_bat else "bowling"]
    if stage == "await_resume":
        ao = g.get("active_over") or {}
        already_ready = ao.get("resume_batting_ready" if i_bat else "resume_bowling_ready", False)
        if already_ready:
            raise ValueError("Too late — you've already confirmed you're ready to resume.")
    elif i_bat:
        if stage not in ("play", "await_batter"):
            raise ValueError("Can't make an Impact Player swap right now.")
        if stage == "play" and my_pending["submitted"]:
            raise ValueError("Too late — you've already locked in this over.")
    else:
        if stage != "play":
            raise ValueError("Can't make an Impact Player swap right now.")
        if my_pending["submitted"]:
            raise ValueError("Too late — you've already locked in this over.")

    xi = g["teams"][role]["xi"]
    xi_names = [p["name"] for p in xi]
    if out_name not in xi_names:
        raise ValueError("That player isn't in your XI.")
    squad_by_name = {p["name"]: p for p in g["squads"][role]["roster"]}
    if in_name not in squad_by_name or in_name in xi_names:
        raise ValueError("That player isn't available on your bench.")
    if in_name == out_name:
        raise ValueError("Pick a different player to bring on.")

    st = g["state"]
    striker = st.get_striker() if i_bat else None
    non_striker = st.get_non_striker() if i_bat else None
    is_striker = i_bat and striker is not None and striker.name == out_name
    is_non_striker = i_bat and non_striker is not None and non_striker.name == out_name
    # Any of the 11 can be replaced -- including one who's already out (they
    # can still be swapped for a fresh bowling option, since dismissal only
    # ends a player's batting, not their standing in the XI). The one thing
    # that's NOT allowed: at await_batter, out_name can't be whoever's still
    # live in the OTHER crease slot -- that slot isn't part of this action
    # (it's not vacant), so swapping them here would corrupt the XI/lineup.
    if i_bat and stage == "await_batter" and (is_striker or is_non_striker):
        raise ValueError("That batter is still in — pick someone else to replace.")

    # squad-composition guard on the resulting XI (same rule as locking one)
    new_xi = [dict(p) for p in xi]
    idx = next(i for i, p in enumerate(new_xi) if p["name"] == out_name)
    new_xi[idx] = _card_fields(BY_NAME[in_name])
    os_in = sum(1 for p in new_xi if p.get("is_foreigner"))
    wk_in = sum(1 for p in new_xi if p.get("is_keeper"))
    if os_in > XI_MAX_OVERSEAS:
        raise ValueError(f"Max {XI_MAX_OVERSEAS} overseas players.")
    if wk_in < 1:
        raise ValueError("Your XI needs at least 1 wicket-keeper.")

    g["teams"][role]["xi"] = new_xi

    if i_bat:
        new_batter = _make_batter(BY_NAME[in_name])
        if stage == "await_batter":
            # Fills the open crease slot by replacing out_name's slot IN
            # PLACE (not appended -- keeps len(lineup) fixed at the original
            # XI size, same pattern as every other case below). out_name's
            # array slot is just a pointer target, safe to reuse whether they
            # were dismissed already or never batted -- their own bat_card
            # row is keyed by name and stays untouched either way (see the
            # guard above, which only rules out the OTHER still-live slot).
            # Then runs the SAME post-fill bookkeeping set_next_batter does:
            # whether the game moves to
            # 'play', 'await_resume', or straight into finishing the over
            # depends on whether one was already mid-flight when the wicket
            # fell. Omitting this left the game stuck in 'await_batter'
            # forever -- the crease was filled internally, but nothing ever
            # advanced the stage, so neither side ever saw a way to proceed.
            lu_idx = next(i for i, b in enumerate(st.lineup) if b.name == out_name)
            st.lineup[lu_idx] = new_batter
            if g.get("vacant_slot") == "non_striker":
                st.non_striker_index = lu_idx
            else:
                st.striker_index = lu_idx
            g["used_batters"].append(in_name)
            _ensure_bat_row(in_name)

            if g["active_over"] is None:
                g["stage"] = "play"
            elif st.balls < g["over_end_at"]:
                g["pending_over"]["batting"]["submitted"] = False
                g["pending_over"]["bowling"]["submitted"] = False
                ao = g["active_over"]
                ao["resume_batting_ready"] = False
                ao["resume_bowling_ready"] = False
                g["stage"] = "await_resume"
            else:
                if st.get_striker() is not None:
                    st.rotate_strike()
                _complete_over()
        else:
            lu_idx = next(i for i, b in enumerate(st.lineup) if b.name == out_name)
            st.lineup[lu_idx] = new_batter
            if is_striker or is_non_striker:
                if is_striker:
                    st.striker_index = lu_idx
                else:
                    st.non_striker_index = lu_idx
                g["used_batters"].append(in_name)
                _ensure_bat_row(in_name)
                if out_name in g["bat_card"]:
                    g["bat_card"][out_name]["how_out"] = "retired (Impact sub)"

    ip["used"] = True
    ip["in_name"] = in_name
    ip["out_name"] = out_name
    _emit("", "milestone",
          f"🔄 IMPACT PLAYER — {g['teams'][role]['name']} bring on {in_name} for {out_name}!")


def _finish_over():
    """Called when an over is fully complete: post the end-of-over summary,
    credit the bowler's over, and reset the ready handshake for the next over."""
    g = GAME
    st = g["state"]
    runs_this = st.runs - g["over_start_runs"]
    wkts_this = st.wickets - g["over_start_wickets"]
    over_num = st.balls // 6
    _emit("", "milestone",
          f"End of over {over_num}: {runs_this} run{'s' if runs_this != 1 else ''}, "
          f"{wkts_this} wicket{'s' if wkts_this != 1 else ''}. "
          f"{g['teams'][g['batting_side']]['name']} {st.runs}/{st.wickets}.")
    bname = g["active_over"]["bowler_name"]
    g["bowler_stats"][bname] = g["bowler_stats"].get(bname, 0) + 1
    g["last_bowler"] = bname
    g["active_over"] = None
    _reset_pending()


def _snapshot_innings():
    g = GAME
    st = g["state"]
    g["completed_innings"].append({
        "batting_team": g["batting_side"],
        "batting_team_name": g["teams"][g["batting_side"]]["name"],
        "bowling_team_name": g["teams"][_bowling_side()]["name"],
        "batting_team_color": g["teams"][g["batting_side"]].get("color"),
        "bowling_team_color": g["teams"][_bowling_side()].get("color"),
        "runs": st.runs, "wickets": st.wickets,
        "overs": _overs_str(st.balls), "extras": st.extras,
        "batting": list(g["bat_card"].values()),
        "bowling": list(g["bowl_card"].values()),
    })


def _handle_innings_over():
    g = GAME
    st = g["state"]
    _snapshot_innings()
    if g["innings"] == 1:
        target = st.runs + 1
        g["innings"] = 2
        _prepare_innings(_bowling_side(), target=target, keep_this_over=True)
        g["live"].append({"ball": "", "type": "milestone", "outcome": None,
                          "text": f"End of Innings 1. Target: {target}.", "over": 2})
    else:
        inn1, inn2 = g["completed_innings"][0], g["completed_innings"][1]
        scores, _, _ = _fixture_impact_scores(inn1, inn2)
        g["motm"] = max(scores, key=scores.get) if scores else None
        _compute_result()
        g["phase"] = "finished"
        g["stage"] = "done"
        if g.get("tournament"):
            _finish_tournament_fixture()


def _compute_result():
    g = GAME
    st = g["state"]
    batting = g["batting_side"]
    bowling = _bowling_side()
    chasing = g["teams"][batting]["name"]
    defending = g["teams"][bowling]["name"]
    target = g["target"]
    if st.runs >= target:
        wkts_left = min(10, len(st.lineup) - 1) - st.wickets
        g["result"] = f"{chasing} won by {wkts_left} wicket{'s' if wkts_left != 1 else ''}!"
        g["match_winner"] = batting
    elif st.runs == target - 1:
        g["result"] = "Match tied!"
        g["match_winner"] = None
    else:
        margin = target - 1 - st.runs
        g["result"] = f"{defending} won by {margin} run{'s' if margin != 1 else ''}!"
        g["match_winner"] = bowling


def _try_resolve_over():
    """If both sides have submitted, lock intents and simulate the over."""
    g = GAME
    p = g["pending_over"]
    if not (p["batting"]["submitted"] and p["bowling"]["submitted"]):
        return

    st = g["state"]
    striker, non_striker = st.get_striker(), st.get_non_striker()
    g["active_over"] = {
        "bowler_name": p["bowling"]["bowler_name"],
        "bowl_intent": p["bowling"]["bowl_intent"],
        "bowl_role": p["bowling"].get("bowl_role", DEFAULT_BOWL_ROLE),
        "striker_intent": p["batting"]["striker_intent"],
        "non_striker_intent": p["batting"]["non_striker_intent"],
        "striker_role": p["batting"].get("striker_role", DEFAULT_BAT_ROLE),
        "non_striker_role": p["batting"].get("non_striker_role", DEFAULT_BAT_ROLE),
        # locked in BY NAME, not by crease slot, so a mid-over strike rotation
        # (or a free-hit ball that itself rotates strike) can never cause the
        # two batters' intents/roles to get swapped onto the wrong person.
        "intent_by_name": {
            **({striker.name: p["batting"]["striker_intent"]} if striker else {}),
            **({non_striker.name: p["batting"]["non_striker_intent"]} if non_striker else {}),
        },
        "role_by_name": {
            **({striker.name: p["batting"].get("striker_role", DEFAULT_BAT_ROLE)} if striker else {}),
            **({non_striker.name: p["batting"].get("non_striker_role", DEFAULT_BAT_ROLE)} if non_striker else {}),
        },
        # one-shot gambits, armed secretly with the submissions and consumed here
        "attack_gambit": bool(p["batting"].get("gambit")),
        "trap_gambit": bool(p["bowling"].get("gambit")),
    }
    g["bowler"] = _make_bowler(BY_NAME[p["bowling"]["bowler_name"]], p["bowling"]["bowl_intent"])
    g["this_over"] = []
    g["over_end_at"] = st.balls + (6 - st.balls % 6 if st.balls % 6 else 6)
    g["over_start_runs"] = st.runs
    g["over_start_wickets"] = st.wickets

    # consume + reveal gambits at over start (both sides are committed by now,
    # so announcing them here leaks no decision-relevant information)
    cards = g.get("gambit_cards") or {}
    if g["active_over"]["attack_gambit"]:
        cards.get(g["batting_side"], {})["attack"] = False
        _emit("", "milestone", f"⚡ GAMBIT — {g['teams'][g['batting_side']]['name']} go ALL OUT ATTACK this over!")
    if g["active_over"]["trap_gambit"]:
        cards.get(_bowling_side(), {})["trap"] = False
        _emit("", "milestone", f"⚡ GAMBIT — {g['teams'][_bowling_side()]['name']} set a TRAP this over!")

    _run_and_route()


def _complete_over():
    """Finish the current over, then either end the innings (over limit reached)
    or open the next over. Owns the stage transition so callers don't clobber it."""
    _finish_over()
    if GAME["state"].balls >= OVERS_PER_INNINGS * 6:
        _handle_innings_over()   # -> stage 'openers' (innings 2) or 'done' (finished)
    else:
        GAME["stage"] = "play"


def _begin_free_hit():
    ao = GAME["active_over"]
    GAME["free_hit"] = {
        "active": True, "batting_ready": False, "bowling_ready": False,
        "striker_intent": ao["striker_intent"],
        "non_striker_intent": ao["non_striker_intent"],
        "bowl_intent": ao["bowl_intent"],
        "striker_role": ao.get("striker_role", DEFAULT_BAT_ROLE),
        "non_striker_role": ao.get("non_striker_role", DEFAULT_BAT_ROLE),
        "bowl_role": ao.get("bowl_role", DEFAULT_BOWL_ROLE),
    }
    GAME["stage"] = "free_hit"


def _run_and_route():
    status = _simulate_until_pause()
    if status == "innings_over":
        _handle_innings_over()
    elif status == "wicket_pending":
        GAME["stage"] = "await_batter"
    elif status == "free_hit_pending":
        _begin_free_hit()
    else:  # over_complete
        _complete_over()
    _bump()


# --- Auction ------------------------------------------------------------------

def _new_squad():
    return {"budget": INITIAL_PURSE, "roster": [], "os": 0, "wk": 0, "locked": False}


def _start_auction():
    per_set = auction_pool_size_per_set(len(GAME["team_ids"]))
    sets, _total, _os = generate_draft_pool(ALL_PLAYERS, players_per_set=per_set)
    pool = []
    for s in sets:
        for p in s["players"]:
            pool.append({**_card_fields(p), "tier": s["tier"], "role": s["role"], "set_id": s["set_id"]})
    # The "Available Players" reference list is sorted alphabetically per set —
    # deliberately decoupled from `s["players"]`'s own (separately shuffled)
    # presentation order, so who's up next in bidding can't be read off the list.
    pool.sort(key=lambda c: c["name"])
    GAME["squads"] = {t: _new_squad() for t in GAME["team_ids"]}
    GAME["auction"] = {
        "sets": sets, "pool": pool, "set_index": 0, "player_index": 0,
        "stage": "preview",           # preview | bidding | resolved | done
        "current": None, "current_bid": BASE_BID, "active_bidder": None, "strike": 0,
        "deadline": None, "total_wait": 0.0,
        "out": {t: False for t in GAME["team_ids"]},        # folded on the CURRENT lot
        "ready": {t: False for t in GAME["team_ids"]},
        "skip_votes": {t: False for t in GAME["team_ids"]},  # want to skip the rest of THIS set
        "message": "Review the full player pool. All teams press Ready to begin.",
        "unsold": [],
    }
    GAME["phase"] = "auction"


def _cur_set():
    a = GAME["auction"]
    return a["sets"][a["set_index"]] if a["set_index"] < len(a["sets"]) else None


def _present_player():
    a = GAME["auction"]
    s = _cur_set()
    a["current"] = s["players"][a["player_index"]]
    a["current_bid"] = TIER_OPENING.get(s["tier"], BASE_BID)
    a["active_bidder"] = None
    a["strike"] = 0
    # locked squads auto-pass every remaining lot
    a["out"] = {t: GAME["squads"][t]["locked"] for t in GAME["team_ids"]}
    a["ready"] = {t: False for t in GAME["team_ids"]}
    a["stage"] = "bidding"
    a["message"] = f"{a['current']['name']} is up — {s['tier']} {s['role']}. Opening at ₹{a['current_bid']:.1f} Cr."
    _set_deadline(s["tier"])


def _set_deadline(tier):
    a = GAME["auction"]
    wait = TIER_WAIT.get(tier, 2.5) * (0.8 + random.random() * 0.4)
    a["total_wait"] = wait
    a["deadline"] = time.time() + wait


def _advance_lot():
    """Present the next lot (or finish the auction). No auto set-break — the
    ready-gate lives in the 'resolved' stage between lots."""
    a = GAME["auction"]
    a["player_index"] += 1
    s = _cur_set()
    if s and a["player_index"] >= len(s["players"]):
        a["set_index"] += 1
        a["player_index"] = 0
        s = _cur_set()
        _reset_skip_votes()
    if s is None:
        _auction_done()
    else:
        _present_player()


def _reset_skip_votes():
    GAME["auction"]["skip_votes"] = {t: False for t in GAME["team_ids"]}


def _skip_would_strand_squads():
    """Would skipping the rest of THIS set make it mathematically impossible
    for every still-active team to reach a legal squad? There's no auto-fill
    safety net, so once the numbers don't add up, someone is guaranteed to
    get kicked out at the grace period with no way to prevent it -- players
    should see that coming, not discover it after the fact.

    Two ways a skip can strand a team:
    1. Not enough total players left afterward for everyone's outstanding
       SQUAD_MIN need combined (ignores that teams also compete for the same
       players, so it's a necessary-not-sufficient check -- but it catches
       the worst case, which is what actually burned someone here).
    2. This is the LAST Wicket Keeper set left in the pool and some active
       team still has zero keepers -- total headcount could be fine while
       the keeper requirement becomes impossible to ever satisfy."""
    a = GAME["auction"]
    remaining_sets = a["sets"][a["set_index"] + 1:]
    active = [t for t in GAME["team_ids"] if not GAME["squads"][t]["locked"]]
    if not active:
        return False

    remaining_supply = sum(len(s["players"]) for s in remaining_sets)
    need = sum(max(0, SQUAD_MIN - len(GAME["squads"][t]["roster"])) for t in active)
    if remaining_supply < need:
        return True

    keeper_sets_left = any(s["role"] == "Wicket Keeper" for s in remaining_sets)
    cur = _cur_set()
    is_last_keeper_set = (cur is not None and cur["role"] == "Wicket Keeper" and not keeper_sets_left)
    if is_last_keeper_set and any(GAME["squads"][t]["wk"] < 1 for t in active):
        return True

    return False


def _skip_to_next_set():
    """Every active team agreed to skip whatever's left of the current set.
    Whatever lot is currently up (if any) just goes unsold -- everyone
    consented to move on rather than deal with it."""
    a = GAME["auction"]
    if a["current"]:
        a["unsold"].append(_card_fields(a["current"]))
    a["set_index"] += 1
    a["player_index"] = 0
    _reset_skip_votes()
    s = _cur_set()
    if s is None:
        _auction_done()
    else:
        _present_player()


def _resolve(stage_msg, sold=False):
    """Move a finished lot into the ready-gated 'resolved' stage (no timer runs
    until BOTH teams press Ready for the next lot)."""
    a = GAME["auction"]
    a["stage"] = "resolved"
    a["deadline"] = None
    a["ready"] = {t: False for t in GAME["team_ids"]}
    a["message"] = stage_msg
    a["last_result"] = "sold" if sold else "unsold"


def _execute_sale():
    a = GAME["auction"]
    w = a["active_bidder"]
    sq = GAME["squads"][w]
    p = a["current"]
    role = _cur_set()["role"]
    if len(sq["roster"]) >= SQUAD_MAX:
        return _execute_unsold()   # squad full, can't buy
    sq["budget"] = round(sq["budget"] - a["current_bid"], 1)
    sq["roster"].append({**_card_fields(p), "assigned_role": role, "price": a["current_bid"]})
    if p.get("is_foreigner"):
        sq["os"] += 1
    if p.get("is_keeper") or role == "Wicket Keeper":
        sq["wk"] += 1
    _resolve(f"SOLD! {p['name']} to {GAME['teams'][w]['name']} for ₹{a['current_bid']:.1f} Cr!", sold=True)


def _execute_unsold():
    a = GAME["auction"]
    a["unsold"].append(_card_fields(a["current"]))
    _resolve(f"UNSOLD! {a['current']['name']} finds no takers.", sold=False)


AUCTION_GRACE_SECONDS = 60.0

def _auction_done():
    a = GAME["auction"]
    a["stage"] = "done"
    a["current"] = None
    # grace-period deadline: any squad still short of SQUAD_MIN when this
    # expires auto-forfeits (see _check_auction_grace_expiry).
    a["deadline"] = time.time() + AUCTION_GRACE_SECONDS
    a["total_wait"] = AUCTION_GRACE_SECONDS
    a["message"] = f"All lots done! Fill up and lock in your squad within {int(AUCTION_GRACE_SECONDS)}s."


def _check_auction_grace_expiry():
    """Fired by the ticker once the post-auction grace deadline passes. Any
    squad still under SQUAD_MIN (or without a keeper) is kicked out — no
    auto-fill safety net.

    Plain 1v1 games: reuses the same 'abandoned' result path as a manual
    /api/exit_game forfeit, so _serialize's redaction/finished-banner handling
    just works unchanged. Tournament games: the short squad(s) are removed
    from team_ids entirely via _eliminate_from_auction, before any fixtures
    are generated, so the round robin is simply built among whoever's left."""
    a = GAME["auction"]
    team_ids = GAME["team_ids"]
    losers = [t for t in team_ids if not _squad_valid(t)]
    a["deadline"] = None
    if not losers:
        return   # everyone qualified after all; nothing to enforce

    if GAME.get("tournament"):
        for t in losers:
            _eliminate_from_auction(t)
        remaining = GAME["team_ids"]
        if len(remaining) < 2:
            GAME["abandoned"] = True
            GAME["abandoned_by"] = "__draw__"
            GAME["result"] = "Tournament abandoned — not enough squads reached the minimum."
            return
        if all(GAME["squads"][t]["locked"] for t in remaining):
            _to_grounds()
        return

    if len(losers) == len(team_ids):
        GAME["abandoned"] = True
        GAME["abandoned_by"] = "__draw__"
        GAME["result"] = "Match abandoned — neither team completed a squad in time."
    else:
        loser = losers[0]
        winner = next(t for t in team_ids if t != loser)
        GAME["abandoned"] = True
        GAME["abandoned_by"] = loser
        GAME["result"] = f"{GAME['teams'][winner]['name']} wins — {GAME['teams'][loser]['name']} failed to complete a squad."


def _process_strike():
    """Fired by the ticker when a BIDDING deadline passes (going once/twice/sold).
    The resolved stage has no deadline, so the auction never auto-advances lots."""
    a = GAME["auction"]
    if a["stage"] != "bidding":
        return
    a["strike"] += 1
    if a["strike"] == 1:
        a["message"] = f"Going once at ₹{a['current_bid']:.1f} Cr..."
        _set_deadline(_cur_set()["tier"])
    elif a["strike"] == 2:
        a["message"] = "Going twice! Last chance!"
        _set_deadline(_cur_set()["tier"])
    else:
        _execute_sale() if a["active_bidder"] else _execute_unsold()


def _auction_tick():
    """Background heartbeat: advances any auction (across ALL concurrent games)
    whose deadline has passed. Tournament fixture advancement is player-driven
    (/api/tournament_ready), not timer-driven, so it doesn't live here.

    Iterates every live game rather than a single global one, since multiple
    unrelated games can now run auctions concurrently. Rebinds the module-level
    GAME to whichever game is being advanced -- same pattern every request
    handler uses -- so _check_auction_grace_expiry/_process_strike/_bump (which
    reference GAME as a bare name) work unchanged."""
    global GAME
    while True:
        time.sleep(0.2)
        with LOCK:
            for g in list(GAMES.values()):
                if g.get("phase") == "auction":
                    a = g["auction"]
                    if a and a["deadline"] and time.time() >= a["deadline"]:
                        GAME = g
                        if a["stage"] == "done":
                            _check_auction_grace_expiry()
                        else:
                            _process_strike()
                        _bump()
            _sweep_stale_games()


def _place_bid(role, amount):
    a = GAME["auction"]
    if a["stage"] != "bidding":
        raise ValueError("No lot is open for bidding.")
    if GAME["squads"][role]["locked"]:
        raise ValueError("Your squad is locked.")
    if a["out"][role]:
        raise ValueError("You have pulled out of this lot.")
    add = float(amount or 0)
    nxt = round(a["current_bid"] + (add if add > 0 else BASE_BID), 1)
    if role == a["active_bidder"] and add <= 0:
        raise ValueError("You already hold the top bid.")
    if GAME["squads"][role]["budget"] < nxt:
        raise ValueError("Not enough purse for that bid.")
    a["active_bidder"] = role
    a["current_bid"] = nxt
    # Once a team pulls out of THIS lot, they stay out for the rest of it --
    # a rival bid must not bring them back in. (out[role] can't already be
    # True here anyway, since the check above rejects bids from a pulled-out
    # team; a["out"] only fully resets on the next lot, see _present_player.)
    a["strike"] = 0
    a["message"] = f"{GAME['teams'][role]['name']} bids ₹{nxt:.1f} Cr!"
    _set_deadline(_cur_set()["tier"])


def _pull_out(role):
    """N-way fold: a fold removes that team from contention for this lot. If
    exactly one un-folded bidder remains and they hold the active bid, sell to
    them immediately; if nobody is left un-folded, the lot goes unsold. This
    generalizes the original 2-bidder shortcut to any number of teams.

    The team currently holding the top bid can't pull out -- otherwise, once
    every rival has folded, the leading bidder could fold too and force the
    lot unsold for free, walking away from a bid they'd otherwise have won."""
    a = GAME["auction"]
    if a["stage"] != "bidding":
        return
    if role == a["active_bidder"]:
        raise ValueError("You hold the top bid — you can't pull out now.")
    a["out"][role] = True
    contenders = [t for t in GAME["team_ids"] if not a["out"][t]]
    if len(contenders) == 1 and a["active_bidder"] == contenders[0]:
        _execute_sale()
    elif len(contenders) == 0:
        _execute_unsold()


def _squad_valid(role):
    sq = GAME["squads"][role]
    n = len(sq["roster"])
    return SQUAD_MIN <= n <= SQUAD_MAX and sq["wk"] >= 1


def _to_xi():
    """Plain 1v1 only -- tournaments skip straight from grounds into fixture 0's
    per-fixture XI pick instead (see _start_tournament_matches / _set_awaiting_xi),
    since they'd otherwise ask for an XI twice in a row for no reason."""
    GAME["phase"] = "xi"
    GAME["xi_select"] = {t: {"xi": [], "locked": False} for t in GAME["team_ids"]}
    # Decide (and reveal) the venue before XI picks, not after, so squads can
    # actually be built around the pitch conditions instead of blind.
    GAME["match_ground"] = _ground_of(random.choice(GAME["team_ids"][:2]))


def _to_grounds():
    """Post-auction home-ground selection: every team claims a UNIQUE stadium
    (server-enforced -- claiming one someone else holds is rejected, so the
    players coordinate in the UI), then locks it. Once all locked -> XI."""
    GAME["phase"] = "grounds"
    GAME["ground_pick"] = {t: {"ground": None, "locked": False} for t in GAME["team_ids"]}


def _resolved_advance():
    """In the 'resolved' ready-gate, advance once every team is ready-or-locked.
    If every team has locked its squad, move on to home-ground selection."""
    a = GAME["auction"]
    sq = GAME["squads"]
    team_ids = GAME["team_ids"]
    def ready(t):
        return a["ready"][t] or sq[t]["locked"]
    if all(ready(t) for t in team_ids):
        if all(sq[t]["locked"] for t in team_ids):
            _to_grounds()
        else:
            _advance_lot()


def _eliminate_from_auction(role):
    """A squad never reached SQUAD_MIN by the post-auction grace deadline:
    kicked out entirely, before any fixtures/XI selection exist yet, so they
    take no further part in this game. Their token stays registered so their
    own client gets a clear "you're out" message (see _serialize) instead of
    a dead/unknown-token error; everyone else's game just continues without
    them, one team_id lighter."""
    GAME["squads"][role]["locked"] = True
    GAME["squads"][role]["eliminated"] = True
    GAME["team_ids"] = [t for t in GAME["team_ids"] if t != role]
    if GAME.get("tournament"):
        t = GAME["tournament"]
        t.setdefault("eliminated", [])
        if role not in t["eliminated"]:
            t["eliminated"].append(role)


def _finalize_xi_to_match():
    """Copy each locked XI into teams[side].xi, then either start the single
    match directly (plain 1v1 game) or kick off the tournament's round robin
    (tournament mode — see _start_next_fixture)."""
    for side in GAME["team_ids"]:
        chosen = GAME["xi_select"][side]["xi"]
        by_name = {p["name"]: p for p in GAME["squads"][side]["roster"]}
        GAME["teams"][side]["xi"] = [
            {"name": n, "batting_ovr": by_name[n]["batting_ovr"],
             "bowling_ovr": by_name[n]["bowling_ovr"],
             "is_foreigner": by_name[n].get("is_foreigner", False),
             "is_keeper": by_name[n].get("is_keeper", False)}
            for n in chosen
        ]
    if GAME.get("tournament"):
        _start_tournament_matches()
    else:
        GAME["match_teams"] = GAME["team_ids"][:2]
        # match_ground was already decided (and shown to both teams) back in
        # _to_xi(), before XI selection started -- don't re-roll it here.
        _start_single_match()


def _ground_of(team_id):
    """The stadium dict a team claimed in the grounds phase (random fallback
    for paths that never ran it, e.g. Quick Match)."""
    gp = GAME.get("ground_pick") or {}
    gid = (gp.get(team_id) or {}).get("ground")
    return STADIUM_BY_ID.get(gid) or random.choice(STADIUMS)


def _ground_view():
    """Public serialization of the current match's stadium."""
    mg = GAME.get("match_ground")
    if not mg:
        return None
    return {"name": mg["name"], "city": mg["city"], "pitch": mg["pitch"],
            "pitch_desc": PITCH_DESCRIPTIONS.get(mg["pitch"], "")}


def _start_single_match():
    GAME["innings"] = 1
    GAME["completed_innings"] = []
    GAME["live"] = []
    GAME["motm"] = None
    if not GAME.get("match_ground"):
        GAME["match_ground"] = random.choice(STADIUMS)
    # one-shot gambit cards, per match per team: All Out Attack is played
    # while batting, Trap Set while bowling (see conditions.py for effects)
    GAME["gambit_cards"] = {t: {"attack": True, "trap": True} for t in GAME["match_teams"]}
    # Impact Player: one bench-for-XI swap per team, once for the whole match
    # (usable whichever innings/role they're in when they choose to spend it)
    GAME["impact_player"] = {t: {"used": False, "in_name": None, "out_name": None}
                              for t in GAME["match_teams"]}
    GAME["phase"] = "match"
    _do_toss()


# --- Tournament (3-8 teams): shared auction already ran; this drives the ------
# --- round robin -> Qualifier 1 -> Qualifier 2 -> Final sequence -------------

def _blank_standing():
    return {"played": 0, "won": 0, "lost": 0, "points": 0,
            "runs_for": 0, "overs_for": 0.0, "runs_against": 0, "overs_against": 0.0, "nrr": 0.0}


def _overs_to_float(overs_str, all_out=False):
    """Standard cricket NRR convention: an all-out innings counts as the full
    quota of overs, regardless of how many balls it actually took."""
    if all_out:
        return float(OVERS_PER_INNINGS)
    whole, _, balls = str(overs_str).partition(".")
    return int(whole or 0) + int(balls or 0) / 6.0


def _ranked_teams():
    """Team ids sorted by tournament points, then Net Run Rate, both descending."""
    t = GAME["tournament"]
    return sorted(GAME["team_ids"],
                  key=lambda x: (-t["standings"][x]["points"], -t["standings"][x]["nrr"]))


def _new_fixture(a, b, kind):
    return {"a": a, "b": b, "kind": kind, "played": False, "winner": None,
            "result_text": None, "motm": None}


def _fixture_impact_scores(inn1, inn2):
    """Combines batting + bowling performances across both innings of a
    fixture into one impact score per player: runs + wickets*20 (a common
    fantasy-cricket-style weighting) -- used for Man of the Match and the
    tournament-wide MVP award. Returns (scores, per_player_runs, per_player_wkts)."""
    scores, runs_by, wkts_by = {}, {}, {}
    for inn in (inn1, inn2):
        for row in inn["batting"]:
            scores[row["name"]] = scores.get(row["name"], 0) + row["runs"]
            runs_by[row["name"]] = runs_by.get(row["name"], 0) + row["runs"]
        for row in inn["bowling"]:
            scores[row["name"]] = scores.get(row["name"], 0) + row["wickets"] * 20
            wkts_by[row["name"]] = wkts_by.get(row["name"], 0) + row["wickets"]
    return scores, runs_by, wkts_by


def _record_fixture_player_stats(t, a, b, inn1, inn2, motm_name):
    """Fold one fixture's batting/bowling cards into the tournament-long
    player_stats accumulator (Orange/Purple Cap, MVP), and bump the Man of
    the Match's tally."""
    stats = t.setdefault("player_stats", {})
    for inn in (inn1, inn2):
        batting_team = inn["batting_team"]
        bowling_team = b if batting_team == a else a
        for row in inn["batting"]:
            s = stats.setdefault(row["name"], {"runs": 0, "wickets": 0, "motm": 0, "team": batting_team})
            s["runs"] += row["runs"]
            s["team"] = batting_team
        for row in inn["bowling"]:
            s = stats.setdefault(row["name"], {"runs": 0, "wickets": 0, "motm": 0, "team": bowling_team})
            s["wickets"] += row["wickets"]
            s["team"] = bowling_team
    if motm_name and motm_name in stats:
        stats[motm_name]["motm"] += 1


AWARDS_LEADERBOARD_SIZE = 20

def _tournament_awards(t, top_n=AWARDS_LEADERBOARD_SIZE):
    """Orange Cap (most runs), Purple Cap (most wickets), MVP (best combined
    impact score) leaderboards across the whole tournament -- top `top_n`
    each, not just the single leader. None if nobody's played yet."""
    stats = t.get("player_stats") or {}
    if not stats:
        return None
    def leaderboard(key_fn):
        ranked = sorted(stats.keys(), key=key_fn, reverse=True)[:top_n]
        return [{"name": n, "team": stats[n]["team"], "value": key_fn(n)} for n in ranked]
    orange = leaderboard(lambda n: stats[n]["runs"])
    purple = leaderboard(lambda n: stats[n]["wickets"])
    mvp = leaderboard(lambda n: stats[n]["runs"] + stats[n]["wickets"] * 20)
    return {"orange_cap": orange, "purple_cap": purple, "mvp": mvp}


def _round_robin_pairs(team_ids):
    """Standard circle-method round-robin scheduling: every unique pair once,
    ordered round by round (each round plays every team at most once) so no
    single team is stuck playing several fixtures back-to-back before the
    others get a turn -- fixtures still resolve one at a time in this list
    order, but the ORDER now spreads each team's matches out realistically."""
    teams = list(team_ids)
    bye = object() if len(teams) % 2 == 1 else None
    if bye is not None:
        teams.append(bye)
    n = len(teams)
    rounds = []
    for _ in range(n - 1):
        round_pairs = []
        for i in range(n // 2):
            a, b = teams[i], teams[n - 1 - i]
            if a is not bye and b is not bye:
                round_pairs.append((a, b))
        rounds.append(round_pairs)
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return [pair for rnd in rounds for pair in rnd]


def _start_tournament_matches():
    """Called once every team has locked its home ground: build the
    round-robin fixture list (every unique pair, once, realistically ordered
    -- see _round_robin_pairs), then send fixture 0's two teams straight into
    picking their XI for it via the SAME per-fixture flow every later fixture
    uses (_set_awaiting_xi) -- there's no separate one-off XI pick right after
    the auction that just gets asked again before the real first match."""
    team_ids = GAME["team_ids"]
    fixtures = [_new_fixture(a, b, "round_robin") for a, b in _round_robin_pairs(team_ids)]
    GAME["tournament"].update({
        "fixtures": fixtures, "current_fixture": 0, "stage": "round_robin",
        "standings": {t: _blank_standing() for t in team_ids},
        "champion": None, "eliminated": [],
        "awaiting_ready": False, "next_fixture_idx": None, "next_ready": {},
        "fatigue": {},   # player name -> consecutive matches played without a rest
    })
    _set_awaiting_xi(0)


def _start_fixture(idx):
    g = GAME
    fx = g["tournament"]["fixtures"][idx]
    g["tournament"]["current_fixture"] = idx
    g["match_teams"] = [fx["a"], fx["b"]]
    g["abandoned"] = False
    g["abandoned_by"] = None
    g["result"] = None
    g["match_winner"] = None
    # match_ground for this fixture was already decided (and shown to both
    # teams during XI selection) by _set_awaiting_xi / _to_xi -- don't re-roll
    # it here, or the venue shown pre-match wouldn't match the one actually
    # played on. Fallback covers paths that skip that step (e.g. tests).
    if not g.get("match_ground"):
        g["match_ground"] = _ground_of(fx["a"]) if fx["kind"] == "round_robin" else random.choice(STADIUMS)
    _start_single_match()


def _finish_tournament_fixture():
    """Called right after _compute_result() for a tournament fixture: folds the
    match into the standings/NRR, then immediately works out what's next (see
    _determine_next_fixture) — but does NOT start it. Whoever's up next has to
    explicitly press Ready (/api/tournament_ready); nobody is ever auto-swept
    into a new match or auto-kicked anywhere."""
    g = GAME
    t = g["tournament"]
    fx_idx = t["current_fixture"]
    fx = t["fixtures"][fx_idx]
    a, b = g["match_teams"]
    inn1, inn2 = g["completed_innings"][0], g["completed_innings"][1]
    by_team = {inn1["batting_team"]: inn1, inn2["batting_team"]: inn2}
    fx["motm"] = g.get("motm")
    _record_fixture_player_stats(t, a, b, inn1, inn2, fx["motm"])
    # _start_single_match() wipes GAME["completed_innings"] the moment the next
    # fixture kicks off, so archive this one here or it's gone for good.
    t.setdefault("fixture_scorecards", {})[fx_idx] = [inn1, inn2]

    # Energy bookkeeping: everyone who played this fixture tires a notch
    # (-0.5% effective OVR per consecutive match, see _energy_mult); squad
    # members who sat this one out come back fully refreshed.
    fatigue = t.setdefault("fatigue", {})
    for side in (a, b):
        xi_names = {p["name"] for p in g["teams"][side]["xi"]}
        for p in (g.get("squads") or {}).get(side, {}).get("roster", []):
            if p["name"] in xi_names:
                fatigue[p["name"]] = fatigue.get(p["name"], 0) + 1
            else:
                fatigue[p["name"]] = 0

    # Points-table standings are a round-robin-only concept in real
    # tournaments -- playoff fixtures (qualifier1/qualifier2/final) decide the
    # champion directly and must NOT inflate a team's played/won/points/NRR.
    if fx["kind"] == "round_robin":
        for team_id in (a, b):
            row = t["standings"][team_id]
            mine = by_team[team_id]
            theirs = by_team[b if team_id == a else a]
            row["runs_for"] += mine["runs"]
            row["overs_for"] += _overs_to_float(mine["overs"], mine["wickets"] >= 10)
            row["runs_against"] += theirs["runs"]
            row["overs_against"] += _overs_to_float(theirs["overs"], theirs["wickets"] >= 10)
            row["played"] += 1
            row["nrr"] = ((row["runs_for"] / row["overs_for"] if row["overs_for"] else 0.0)
                          - (row["runs_against"] / row["overs_against"] if row["overs_against"] else 0.0))

    fx["played"] = True
    fx["result_text"] = g["result"]
    winner = g.get("match_winner")
    if fx["kind"] == "round_robin":
        if winner:
            loser = b if winner == a else a
            fx["winner"] = winner
            t["standings"][winner]["won"] += 1
            t["standings"][winner]["points"] += 2
            t["standings"][loser]["lost"] += 1
        else:
            t["standings"][a]["points"] += 1
            t["standings"][b]["points"] += 1
    elif winner:
        # a playoff fixture won outright on the field -- _playoff_winner (in
        # _determine_next_fixture, called below) only needs to step in for a
        # genuine tie, which doesn't touch standings either way
        fx["winner"] = winner

    _determine_next_fixture()


def _set_awaiting_ready(idx):
    """Park the tournament between fixtures: the two teams in fixture `idx`
    must both press Ready (/api/tournament_ready) before it actually starts."""
    t = GAME["tournament"]
    fx = t["fixtures"][idx]
    t["next_fixture_idx"] = idx
    t["next_ready"] = {fx["a"]: False, fx["b"]: False}
    t["awaiting_ready"] = True


def _seed_playoffs():
    """Seeds the top-3 into Qualifier 1. Auction-time elimination (see
    _eliminate_from_auction) can leave a tournament with fewer teams than it
    started, so this degrades gracefully instead of assuming 3+ survivors:
    2 teams left -> straight to a final (skip the qualifier bracket entirely);
    1 (or 0) left -> whoever's standing is simply the champion, no match needed."""
    t = GAME["tournament"]
    ranked = _ranked_teams()
    if len(ranked) < 2:
        t["champion"] = ranked[0] if ranked else None
        t["stage"] = "champion"
        return
    if len(ranked) == 2:
        t["playoff_seeds"] = ranked[:2]
        t["stage"] = "final"
        t["fixtures"].append(_new_fixture(ranked[0], ranked[1], "final"))
        _set_awaiting_ready(len(t["fixtures"]) - 1)
        return
    top3 = ranked[:3]
    t["playoff_seeds"] = top3
    t["stage"] = "qualifier1"
    t["fixtures"].append(_new_fixture(top3[0], top3[1], "qualifier1"))
    _set_awaiting_ready(len(t["fixtures"]) - 1)


def _playoff_winner(fx):
    """A playoff fixture MUST produce a single team to carry forward — there's
    no Super Over engine here, so a tied Qualifier/Final falls back to whoever
    finished higher in the round-robin standings (a common real-world
    tiebreak when a decider can't be replayed)."""
    if fx["winner"]:
        return fx["winner"]
    ranked = _ranked_teams()
    winner = fx["a"] if ranked.index(fx["a"]) < ranked.index(fx["b"]) else fx["b"]
    fx["winner"] = winner
    fx["result_text"] = (fx.get("result_text") or "Match tied") + \
        f" — {GAME['teams'][winner]['name']} advance on standings."
    return winner


def _determine_next_fixture():
    """Called immediately once a fixture's result is recorded: works out the
    next pairing (or crowns a champion) and parks it behind the ready-gate.
    Never starts a match itself — that only happens once both of the next
    fixture's teams call /api/tournament_ready."""
    g = GAME
    t = g["tournament"]

    if t["stage"] == "round_robin":
        pending = [i for i, f in enumerate(t["fixtures"]) if not f["played"]]
        if pending:
            _set_awaiting_ready(pending[0])
        else:
            _seed_playoffs()
        return

    if t["stage"] == "qualifier1":
        fx = t["fixtures"][t["current_fixture"]]
        winner = _playoff_winner(fx)
        t["q1_winner"] = winner
        q1_loser = fx["b"] if winner == fx["a"] else fx["a"]
        third = t["playoff_seeds"][2]
        t["stage"] = "qualifier2"
        t["fixtures"].append(_new_fixture(q1_loser, third, "qualifier2"))
        _set_awaiting_ready(len(t["fixtures"]) - 1)
        return

    if t["stage"] == "qualifier2":
        fx = t["fixtures"][t["current_fixture"]]
        q2_winner = _playoff_winner(fx)
        t["stage"] = "final"
        t["fixtures"].append(_new_fixture(t["q1_winner"], q2_winner, "final"))
        _set_awaiting_ready(len(t["fixtures"]) - 1)
        return

    if t["stage"] == "final":
        fx = t["fixtures"][t["current_fixture"]]
        t["champion"] = _playoff_winner(fx)
        t["stage"] = "champion"
        return


def _confirm_tournament_ready(role):
    """A team presses Ready for the fixture the ready-gate is holding on. Once
    both of that fixture's teams have confirmed, move on to picking a fresh
    XI for this specific fixture (see _set_awaiting_xi) rather than starting
    the match immediately -- squads don't replay the same XI all tournament."""
    t = GAME["tournament"]
    if not t.get("awaiting_ready") or role not in t.get("next_ready", {}):
        raise ValueError("No fixture is waiting on you right now.")
    t["next_ready"][role] = True
    if all(t["next_ready"].values()):
        idx = t["next_fixture_idx"]
        t["awaiting_ready"] = False
        t["next_fixture_idx"] = None
        t["next_ready"] = {}
        _set_awaiting_xi(idx)


def _set_awaiting_xi(idx):
    """Both teams in fixture `idx` have readied up: they each pick a fresh
    Playing XI from their (already-drafted) squad for this specific match --
    scoped to just these two teams so every other team in the tournament
    keeps seeing the bracket/spectator view undisturbed (GAME["phase"] is
    deliberately left alone here)."""
    t = GAME["tournament"]
    fx = t["fixtures"][idx]
    t["fixture_xi_idx"] = idx
    t["fixture_xi"] = {fx["a"]: {"xi": [], "locked": False}, fx["b"]: {"xi": [], "locked": False}}
    t["awaiting_xi"] = True
    # reveal the venue now, before XI picks -- _start_fixture recomputes the
    # identical value from the same (unchanged) inputs when the match starts.
    if fx["kind"] == "round_robin":
        GAME["match_ground"] = _ground_of(fx["a"])
    else:
        GAME["match_ground"] = random.choice(STADIUMS)


def _lock_fixture_xi(role):
    """One team locks its per-fixture XI. Once both of the fixture's teams
    are locked, copy each into teams[side].xi (same field _prepare_innings
    reads from) and actually start the match."""
    t = GAME["tournament"]
    if not t.get("awaiting_xi") or role not in t.get("fixture_xi", {}):
        raise ValueError("Not selecting an XI right now.")
    sel = t["fixture_xi"][role]
    if len(sel["xi"]) != XI_SIZE:
        raise ValueError(f"Pick exactly {XI_SIZE} players.")
    roster_by_name = {p["name"]: p for p in GAME["squads"][role]["roster"]}
    os_count = sum(1 for n in sel["xi"] if roster_by_name[n].get("is_foreigner"))
    wk_count = sum(1 for n in sel["xi"] if roster_by_name[n].get("is_keeper")
                   or roster_by_name[n].get("assigned_role") == "Wicket Keeper")
    if os_count > XI_MAX_OVERSEAS:
        raise ValueError(f"Max {XI_MAX_OVERSEAS} overseas players.")
    if wk_count < 1:
        raise ValueError("You need at least 1 wicket-keeper.")
    sel["locked"] = True
    idx = t["fixture_xi_idx"]
    fx = t["fixtures"][idx]
    if t["fixture_xi"][fx["a"]]["locked"] and t["fixture_xi"][fx["b"]]["locked"]:
        for side in (fx["a"], fx["b"]):
            chosen = t["fixture_xi"][side]["xi"]
            by_name = {p["name"]: p for p in GAME["squads"][side]["roster"]}
            GAME["teams"][side]["xi"] = [
                {"name": n, "batting_ovr": by_name[n]["batting_ovr"],
                 "bowling_ovr": by_name[n]["bowling_ovr"],
                 "is_foreigner": by_name[n].get("is_foreigner", False),
                 "is_keeper": by_name[n].get("is_keeper", False)}
                for n in chosen
            ]
        t["awaiting_xi"] = False
        t["fixture_xi"] = {}
        _start_fixture(idx)


def _eliminate_from_tournament(role):
    """A team leaves mid-tournament: forfeit only their current fixture (if
    they're playing one right now) and walkover every remaining fixture of
    theirs — the tournament continues for everyone else. (Leaving during the
    shared auction or XI-selection phase, before any fixtures exist yet, is
    a known gap — not handled specially here.)"""
    g = GAME
    t = g["tournament"]
    t.setdefault("eliminated", [])
    if role in t["eliminated"]:
        return
    t["eliminated"].append(role)

    if "fixtures" not in t:
        return   # tournament hasn't reached its first fixture yet

    was_live = g["phase"] == "match" and role in (g.get("match_teams") or [])
    if was_live:
        opp = next(x for x in g["match_teams"] if x != role)
        g["result"] = f"{g['teams'][opp]['name']} wins — {g['teams'][role]['name']} left the tournament."
        g["match_winner"] = opp
        g["phase"] = "finished"
        g["stage"] = "done"
        fx = t["fixtures"][t["current_fixture"]]
        fx["played"] = True
        fx["winner"] = opp
        fx["result_text"] = g["result"]
        t["standings"][opp]["won"] += 1
        t["standings"][opp]["points"] += 2
        t["standings"][opp]["played"] += 1
        t["standings"][role]["lost"] += 1
        t["standings"][role]["played"] += 1

    for fx in t["fixtures"]:
        if not fx["played"] and role in (fx["a"], fx["b"]):
            opp = fx["b"] if fx["a"] == role else fx["a"]
            fx["played"] = True
            fx["winner"] = opp
            fx["result_text"] = f"Walkover — {g['teams'][role]['name']} left the tournament."
            t["standings"][opp]["won"] += 1
            t["standings"][opp]["points"] += 2
            t["standings"][opp]["played"] += 1
            t["standings"][role]["lost"] += 1
            t["standings"][role]["played"] += 1

    if was_live:
        _determine_next_fixture()


def _serialize_spectator_view():
    """Read-only public view of the fixture currently in progress, for
    tournament teams not playing it. Deliberately excludes hidden-intent
    data (pending submissions, bowler pick before reveal) -- only what a
    neutral broadcast viewer would see: score, overs, wickets, ball-by-ball
    commentary for the current over."""
    g = GAME
    if g.get("phase") != "match" or g.get("state") is None or not g.get("batting_side"):
        return None
    st = g["state"]
    batting = g["batting_side"]
    bowling = _bowling_side()
    striker = st.get_striker()
    non_striker = st.get_non_striker()
    bowler = g.get("bowler")
    return {
        "batting_team_name": g["teams"][batting]["name"],
        "bowling_team_name": g["teams"][bowling]["name"],
        "batting_team_color": g["teams"][batting].get("color"),
        "bowling_team_color": g["teams"][bowling].get("color"),
        "innings": g.get("innings"),
        "score": st.runs, "wickets": st.wickets,
        "overs": f"{st.balls // 6}.{st.balls % 6}",
        "target": g.get("target"),
        "striker_name": striker.name if striker else None,
        "non_striker_name": non_striker.name if non_striker else None,
        "bowler_name": bowler.name if bowler else None,
        "this_over": g.get("this_over", []),
        "stage": g.get("stage"),
        "ground": _ground_view(),
    }


def _serialize_tournament_summary():
    """Public bracket/standings view — no hidden-intent data here, safe to show
    to every team including spectators of the fixture currently being played."""
    t = GAME.get("tournament")
    if not t or "fixtures" not in t:
        return None
    standings_view = []
    if t.get("standings"):
        for tid in _ranked_teams():
            standings_view.append({"team_id": tid, "name": GAME["teams"][tid]["name"], **t["standings"][tid]})
    scorecards = t.get("fixture_scorecards") or {}
    fixtures_view = [
        {"idx": i, "a_name": GAME["teams"][f["a"]]["name"], "b_name": GAME["teams"][f["b"]]["name"],
         "kind": f["kind"], "played": f["played"],
         "winner_name": GAME["teams"][f["winner"]]["name"] if f["winner"] else None,
         "result_text": f["result_text"], "motm_name": f.get("motm"),
         "has_scorecard": i in scorecards}
        for i, f in enumerate(t.get("fixtures", []))
    ]
    current = None
    ci = t.get("current_fixture")
    fixtures = t.get("fixtures", [])
    if GAME["phase"] == "match" and ci is not None and 0 <= ci < len(fixtures):
        fx = fixtures[ci]
        current = {"a_name": GAME["teams"][fx["a"]]["name"], "b_name": GAME["teams"][fx["b"]]["name"], "kind": fx["kind"]}
    awards = _tournament_awards(t)
    if awards:
        for key in ("orange_cap", "purple_cap", "mvp"):
            for entry in awards[key]:
                entry["team_name"] = GAME["teams"][entry["team"]]["name"]
    return {
        "stage": t.get("stage"), "standings": standings_view, "fixtures": fixtures_view,
        "current_fixture": current,
        "current_fixture_idx": ci,  # unconditional (unlike `current` above) so the
                                     # client can still find "the fixture I just
                                     # finished" once phase has already moved past "match"
        "champion_name": GAME["teams"][t["champion"]]["name"] if t.get("champion") else None,
        "awards": awards,
    }


# --- Serialization (redacted per role) ---------------------------------------

def _role_of(token):
    return GAME["tokens"].get(token)


def _serialize(token):
    g = GAME
    role = _role_of(token)
    is_tournament = bool(g.get("tournament"))
    # "opponent" (singular) only makes sense for a plain 2-team game
    opp_role = None
    if not is_tournament:
        opp_role = "team2" if role == "team1" else "team1" if role == "team2" else None

    out = {
        "version": g["version"],
        "phase": g["phase"],
        "code": g["code"],
        "is_tournament": is_tournament,
        "you": {"role": role, "joined": role is not None,
                "name": g["teams"][role]["name"] if role else None,
                "color": g["teams"][role].get("color") if role else None},
        "opponent": {
            "joined": g["teams"][opp_role]["joined"] if opp_role else False,
            "name": g["teams"][opp_role]["name"] if opp_role else None,
            "color": g["teams"][opp_role].get("color") if opp_role else None,
        } if opp_role else {"joined": False, "name": None},
        "teams": {t: {"name": g["teams"][t]["name"], "joined": g["teams"][t]["joined"],
                      "color": g["teams"][t].get("color")}
                  for t in g["team_ids"]},
        "color_palette": TEAM_COLOR_PALETTE,
    }
    if is_tournament:
        out["tournament"] = _serialize_tournament_summary()
        t = g["tournament"]
        if role and g.get("squads") and role in g["squads"]:
            fatigue = t.get("fatigue") or {}
            out["my_roster"] = {
                "team_name": g["teams"][role]["name"],
                "players": [
                    {"name": p["name"], "is_foreigner": p.get("is_foreigner", False),
                     "is_keeper": p.get("is_keeper", False),
                     "batting_ovr": p["batting_ovr"], "bowling_ovr": p["bowling_ovr"],
                     "energy": round(max(95.0, 100.0 - 0.5 * fatigue.get(p["name"], 0)), 1),
                     "matches_since_rest": fatigue.get(p["name"], 0)}
                    for p in g["squads"][role]["roster"]
                ],
            }
        if t.get("awaiting_ready") and role in t.get("next_ready", {}):
            nf = t["fixtures"][t["next_fixture_idx"]]
            other = nf["b"] if role == nf["a"] else nf["a"]
            out["next_fixture"] = {
                "a_name": g["teams"][nf["a"]]["name"], "b_name": g["teams"][nf["b"]]["name"],
                "kind": nf["kind"], "i_ready": t["next_ready"][role],
                "opponent_ready": t["next_ready"][other],
            }
        if t.get("awaiting_xi") and role in t.get("fixture_xi", {}):
            idx = t["fixture_xi_idx"]
            fx = t["fixtures"][idx]
            other = fx["b"] if role == fx["a"] else fx["a"]
            sq = g["squads"][role]
            mine = t["fixture_xi"][role]
            chosen = set(mine["xi"])
            fatigue = t.get("fatigue") or {}
            # energy meter: a legible 10%-per-match visual gauge for the
            # rest-or-play decision; the ACTUAL stat penalty is the gentler
            # 0.5%/match shown alongside (see _energy_mult)
            roster = [{**p, "selected": p["name"] in chosen,
                       "energy": max(0, 100 - 10 * fatigue.get(p["name"], 0)),
                       "fatigue_penalty": round(min(5.0, 0.5 * fatigue.get(p["name"], 0)), 1)}
                      for p in sq["roster"]]
            os_in = sum(1 for p in sq["roster"] if p["name"] in chosen and p.get("is_foreigner"))
            wk_in = sum(1 for p in sq["roster"] if p["name"] in chosen
                        and (p.get("is_keeper") or p.get("assigned_role") == "Wicket Keeper"))
            # Same shape as _serialize_xi (fixture-scoped instead of once-per-
            # tournament) so the frontend can reuse renderXI/xiCard unchanged.
            out["fixture_xi"] = {
                "you_role": role, "team_name": g["teams"][role]["name"],
                "opponent_name": g["teams"][other]["name"], "kind": fx["kind"],
                "roster": roster, "xi": mine["xi"], "locked": mine["locked"],
                "opponent_locked": t["fixture_xi"][other]["locked"],
                "others_locked_count": 1 if t["fixture_xi"][other]["locked"] else 0, "others_total": 1,
                "count": len(mine["xi"]), "os": os_in, "wk": wk_in,
                "size": XI_SIZE, "max_os": XI_MAX_OVERSEAS,
                "ground": _ground_view(),
            }
            return out

    if g.get("abandoned"):
        out["abandoned"] = True
        out["ended_result"] = g.get("result") or "Match abandoned."
        if g.get("abandoned_by") == "__draw__":
            out["you_won"] = False
        else:
            out["you_won"] = role is not None and role != g.get("abandoned_by")
        return out

    # squad never reached SQUAD_MIN by the auction grace deadline -- removed
    # from team_ids entirely (see _eliminate_from_auction), so none of the
    # phase-specific branches below (which key off team_ids/xi_select/etc.)
    # are safe to run for this role. Short-circuit with a clear message; the
    # tournament summary above still lets them go watch the bracket.
    if role is not None and (g.get("squads") or {}).get(role, {}).get("eliminated"):
        out["eliminated"] = True
        out["ended_result"] = "Your squad never reached the minimum 15 players in time — you're out."
        out["waiting_for_fixture"] = True
        return out

    if g["phase"] == "lobby":
        if is_tournament:
            t = g["tournament"]
            out["tournament_lobby"] = {
                "size": t["size"],
                "joined_count": sum(1 for tid in g["team_ids"] if g["teams"][tid]["joined"]),
                "roster": [{"team_id": tid, "name": g["teams"][tid]["name"],
                            "joined": g["teams"][tid]["joined"]} for tid in g["team_ids"]],
                "all_joined": all(g["teams"][tid]["joined"] for tid in g["team_ids"]),
                "start_votes": g["start_votes"],
                "i_voted": g["start_votes"].get(role, False) if role else False,
            }
        else:
            out["lobby"] = {
                "start_votes": g["start_votes"],
                "i_voted": g["start_votes"].get(role, False) if role else False,
                "opponent_voted": g["start_votes"].get(opp_role, False) if opp_role else False,
            }
        return out

    if g["phase"] == "auction":
        out["auction"] = _serialize_auction(role)
        return out

    if g["phase"] == "grounds":
        gp = g["ground_pick"]
        out["grounds"] = {
            "stadiums": [{**s, "pitch_desc": PITCH_DESCRIPTIONS[s["pitch"]],
                          "claimed_by": next((g["teams"][t]["name"] for t in g["team_ids"]
                                              if gp[t]["ground"] == s["id"]), None),
                          "claimed_by_me": gp.get(role, {}).get("ground") == s["id"]}
                         for s in STADIUMS],
            "my_ground": gp.get(role, {}).get("ground"),
            "my_locked": gp.get(role, {}).get("locked", False),
            "locked_count": sum(1 for t in g["team_ids"] if gp[t]["locked"]),
            "total_teams": len(g["team_ids"]),
        }
        return out

    if g["phase"] == "xi":
        out["xi"] = _serialize_xi(role)
        return out

    # tournament spectator: this team isn't in the fixture currently being
    # played, so none of the match/toss state below applies to them — show the
    # bracket/standings instead (already attached above as out["tournament"]),
    # plus a read-only public view of the live match if one is in progress.
    if is_tournament and role is not None and role not in (g.get("match_teams") or []):
        out["waiting_for_fixture"] = True
        spectate = _serialize_spectator_view()
        if spectate:
            out["spectate"] = spectate
        return out

    # toss happens before any innings state exists
    if g["stage"] == "toss":
        winner = g["toss"]["winner"]
        out["match"] = {
            "stage": "toss",
            "toss": {"i_won": role == winner, "winner_name": g["teams"][winner]["name"]},
            "ground": _ground_view(),
        }
        return out

    # match / finished
    st = g["state"]
    batting = g["batting_side"]
    bowling = _bowling_side()
    i_bat = role == batting

    def bat_ground(name):
        if not name:
            return None
        r = g["bat_card"].get(name, {})
        rec = BY_NAME[name]
        return {"name": name, "runs": r.get("runs", 0), "balls": r.get("balls", 0),
                "batting_ovr": rec["batting_ovr"], "bowling_ovr": rec["bowling_ovr"],
                **_role_meta(name)}

    striker = st.get_striker()
    non_striker = st.get_non_striker()

    cur_bowler = None
    if g["bowler"] is not None:
        bn = g["bowler"].name
        bc = g["bowl_card"].get(bn, {})
        cur_bowler = {"name": bn, "wickets": bc.get("wickets", 0),
                      "runs": bc.get("runs", 0), "overs": _overs_str(bc.get("balls", 0)),
                      "bowling_ovr": BY_NAME[bn]["bowling_ovr"], "batting_ovr": BY_NAME[bn]["batting_ovr"],
                      **_role_meta(bn)}

    # my bench (role-specific)
    my_bench = []
    if i_bat:
        for p in g["teams"][batting]["xi"]:
            nm = p["name"]
            if striker and nm == striker.name:
                continue
            if non_striker and nm == non_striker.name:
                continue
            row = g["bat_card"].get(nm)
            if row and row["out"]:
                status = "out"
            elif nm in g["used_batters"]:
                status = "out"  # already batted (retired/rotated not modeled) -> unavailable
            else:
                status = "available"
            card = _card_fields(BY_NAME[nm])
            card["status"] = status
            my_bench.append(card)
    else:
        for p in g["teams"][bowling]["xi"]:
            nm = p["name"]
            overs = g["bowler_stats"].get(nm, 0)
            card = _card_fields(BY_NAME[nm])
            card["overs_bowled"] = overs
            card["max_overs"] = MAX_OVERS_PER_BOWLER
            disabled = overs >= MAX_OVERS_PER_BOWLER or nm == g["last_bowler"]
            card["disabled"] = disabled
            my_bench.append(card)

    # opponent list (public: just the other team's XI)
    opp_xi_side = bowling if i_bat else batting
    opponent_list = [
        {"name": p["name"], "batting_ovr": p["batting_ovr"], "bowling_ovr": p["bowling_ovr"],
         **_role_meta(p["name"])}
        for p in g["teams"][opp_xi_side]["xi"]
    ]

    # pending handshake (REDACTED: never expose opponent intents/selection)
    my_pending = g["pending_over"]["batting" if i_bat else "bowling"]
    opp_pending = g["pending_over"]["bowling" if i_bat else "batting"]
    pending = {
        "i_submitted": my_pending["submitted"],
        "opponent_submitted": opp_pending["submitted"],
        "mine": dict(my_pending),
    }
    # Sequenced over: once the bowling side locks in, reveal the bowler's IDENTITY
    # (and public figures) to the batting side — but never the bowl intent.
    if i_bat and opp_pending["submitted"] and opp_pending.get("bowler_name"):
        bn = opp_pending["bowler_name"]
        bc = g["bowl_card"].get(bn, {})
        pending["opponent_bowler"] = {
            "name": bn, "bowling_ovr": BY_NAME[bn]["bowling_ovr"], "batting_ovr": BY_NAME[bn]["batting_ovr"],
            "wickets": bc.get("wickets", 0), "runs": bc.get("runs", 0),
            "overs": _overs_str(bc.get("balls", 0)),
            **_role_meta(bn),
        }

    # free-hit window (REDACTED like the over handshake)
    fh = g["free_hit"]
    my_fh_ready = fh["batting_ready"] if i_bat else fh["bowling_ready"]
    opp_fh_ready = fh["bowling_ready"] if i_bat else fh["batting_ready"]
    free_hit = {
        "active": fh["active"],
        "i_ready": my_fh_ready,
        "opponent_ready": opp_fh_ready,
        "mine": ({"striker_intent": fh["striker_intent"], "non_striker_intent": fh["non_striker_intent"],
                  "striker_role": fh.get("striker_role"), "non_striker_role": fh.get("non_striker_role")}
                 if i_bat else {"bowl_intent": fh["bowl_intent"], "bowl_role": fh.get("bowl_role")}),
    }

    # mid-over resume window after a new batter walks in (REDACTED the same way)
    resume = None
    ao_for_resume = g.get("active_over")
    if g["stage"] == "await_resume" and ao_for_resume:
        resume = {
            "i_ready": ao_for_resume.get("resume_batting_ready" if i_bat else "resume_bowling_ready", False),
            "opponent_ready": ao_for_resume.get("resume_bowling_ready" if i_bat else "resume_batting_ready", False),
        }

    # Impact Player: who's in your XI isn't hidden info, so this is fine to
    # compute the same way for both sides -- only WHEN it's usable differs.
    # Quick Match games have no squad/bench at all (exactly 11 auto-drafted,
    # nothing spare) -- there's simply no one to bring on, so it's unavailable.
    has_bench = bool(g.get("squads")) and role in g["squads"]
    ip_state = (g.get("impact_player") or {}).get(role, {"used": True})
    if g["stage"] == "await_resume":
        already_ready = (ao_for_resume or {}).get("resume_batting_ready" if i_bat else "resume_bowling_ready", False)
        ip_can_use = not already_ready
    elif i_bat:
        ip_can_use = g["stage"] == "await_batter" or (g["stage"] == "play" and not my_pending["submitted"])
    else:
        ip_can_use = g["stage"] == "play" and not my_pending["submitted"]
    ip_can_use = ip_can_use and not ip_state["used"] and has_bench
    ip_pool, ip_out_options = [], []
    if ip_can_use:
        xi_names = {p["name"] for p in g["teams"][role]["xi"]}
        ip_pool = [_card_fields(BY_NAME[p["name"]]) for p in g["squads"][role]["roster"]
                   if p["name"] not in xi_names]
        dismissed = {n for n, row in g["bat_card"].items() if row["out"]} if i_bat else set()
        if i_bat and g["stage"] == "await_batter":
            # Any of the 11 can be swapped out -- including already-dismissed
            # batters (still swappable for a fresh bowling option later) --
            # EXCEPT whoever's still live in the other crease slot right now,
            # since that slot isn't the one this action is filling.
            live_name = striker.name if striker else (non_striker.name if non_striker else None)
            ip_out_options = [{**p, "dismissed": p["name"] in dismissed}
                              for p in g["teams"][role]["xi"] if p["name"] != live_name]
        else:
            ip_out_options = [{**p, "dismissed": p["name"] in dismissed} for p in g["teams"][role]["xi"]]
    impact = {
        "used": ip_state["used"],
        "swap_text": (f"Impact Player used: {ip_state['in_name']} on for {ip_state['out_name']}"
                      if ip_state["used"] and ip_state.get("in_name") else None),
        "can_use": ip_can_use,
        "pool": ip_pool,
        "out_options": ip_out_options,
    }

    out["match"] = {
        "stage": g["stage"],
        "innings": g["innings"],
        "batting_team": batting,
        "bowling_team": bowling,
        "batting_team_name": g["teams"][batting]["name"],
        "bowling_team_name": g["teams"][bowling]["name"],
        "batting_team_color": g["teams"][batting].get("color"),
        "bowling_team_color": g["teams"][bowling].get("color"),
        "i_am_batting": i_bat,
        "runs": st.runs, "wickets": st.wickets, "balls": st.balls,
        "overs": _overs_str(st.balls), "max_overs": OVERS_PER_INNINGS,
        "extras": st.extras, "target": g["target"],
        "striker": bat_ground(striker.name if striker else None),
        "non_striker": bat_ground(non_striker.name if non_striker else None),
        "current_bowler": cur_bowler,
        "await_next_batter": g["stage"] == "await_batter",
        "my_bench": my_bench,
        "opponent_list": opponent_list,
        "this_over": g["this_over"],
        "pending": pending,
        "free_hit": free_hit,
        "resume": resume,
        "result": g["result"],
        "motm": g.get("motm"),
        "ground": _ground_view(),
        "phase_label": phase_for_over(st.balls // 6),
        # own gambit availability + whether one is armed in my pending
        # submission; the opponent's is NEVER exposed pre-resolution (the
        # over-start commentary reveal covers post-resolution)
        "gambits": {
            "available": dict((g.get("gambit_cards") or {}).get(role, {})),
            "i_armed": bool(my_pending.get("gambit")),
        },
        "impact": impact,
    }

    out["live"] = g["live"]
    out["scorecard"] = _serialize_scorecard()
    return out


def _serialize_auction(role):
    a = GAME["auction"]
    sq = GAME["squads"]
    s = _cur_set()
    time_left = max(0.0, a["deadline"] - time.time()) if a["deadline"] else 0.0

    def squad_view(r):
        x = sq[r]
        return {"name": GAME["teams"][r]["name"], "color": GAME["teams"][r].get("color"),
                "budget": round(x["budget"], 1),
                "count": len(x["roster"]), "os": x["os"], "wk": x["wk"],
                "locked": x["locked"], "roster": x["roster"]}

    current = None
    if a["current"]:
        c = a["current"]
        current = {"name": c["name"], "batting_ovr": c["batting_ovr"], "bowling_ovr": c["bowling_ovr"],
                   "is_foreigner": c.get("is_foreigner", False), "is_keeper": c.get("is_keeper", False),
                   "tier": s["tier"] if s else "", "set_role": s["role"] if s else "",
                   **_role_meta(c["name"])}

    def summary_view(r):
        x = sq[r]
        return {"team_id": r, "name": GAME["teams"][r]["name"], "color": GAME["teams"][r].get("color"),
                "budget": round(x["budget"], 1),
                "count": len(x["roster"]), "os": x["os"], "wk": x["wk"], "locked": x["locked"],
                "roster_names": [p["name"] for p in x["roster"]]}

    team_ids = GAME["team_ids"]
    others = [t for t in team_ids if t != role]
    # 2-team back-compat: opp_squad/opp_locked (single opponent, full roster detail)
    opp = others[0] if len(team_ids) == 2 and others else None

    active_teams = [t for t in team_ids if not sq[t]["locked"]]
    skip_votes = a.get("skip_votes", {})

    return {
        "you_role": role, "stage": a["stage"], "message": a["message"],
        "set_id": s["set_id"] if s else None,
        "tier": s["tier"] if s else None, "role_name": s["role"] if s else None,
        "current": current, "current_bid": round(a["current_bid"], 1),
        "active_bidder": a["active_bidder"], "strike": a["strike"],
        "last_result": a.get("last_result"),
        "time_left_ms": int(time_left * 1000), "total_wait_ms": int(a["total_wait"] * 1000),
        "out": a["out"], "ready": a["ready"], "unsold_count": len(a["unsold"]),
        "my_squad": squad_view(role) if role else None,
        "opp_squad": squad_view(opp) if opp else None,
        "my_locked": sq[role]["locked"] if role else False,
        "opp_locked": sq[opp]["locked"] if opp else False,
        "other_squads": [summary_view(t) for t in others] if role else [],
        "can_lock": _squad_valid(role) if role else False,
        "pool": a.get("pool"),
        "squad_min": SQUAD_MIN, "squad_max": SQUAD_MAX, "xi_max_os": XI_MAX_OVERSEAS,
        "skip_set": {
            "i_voted": bool(skip_votes.get(role)) if role else False,
            "count": sum(1 for t in active_teams if skip_votes.get(t)),
            "total": len(active_teams),
            "blocked": _skip_would_strand_squads(),
        },
    }


def _serialize_xi(role):
    xs = GAME["xi_select"]
    sq = GAME["squads"][role]
    mine = xs[role]
    team_ids = GAME["team_ids"]
    others = [t for t in team_ids if t != role]
    chosen = set(mine["xi"])
    roster = [{**p, "selected": p["name"] in chosen} for p in sq["roster"]]
    os_in = sum(1 for p in sq["roster"] if p["name"] in chosen and p.get("is_foreigner"))
    wk_in = sum(1 for p in sq["roster"] if p["name"] in chosen
                and (p.get("is_keeper") or p.get("assigned_role") == "Wicket Keeper"))
    locked_others = sum(1 for t in others if xs[t]["locked"])
    return {
        "you_role": role, "team_name": GAME["teams"][role]["name"],
        "roster": roster, "xi": mine["xi"], "locked": mine["locked"],
        "opponent_locked": xs[others[0]]["locked"] if len(others) == 1 else None,
        "others_locked_count": locked_others, "others_total": len(others),
        "count": len(mine["xi"]), "os": os_in, "wk": wk_in,
        "size": XI_SIZE, "max_os": XI_MAX_OVERSEAS,
        "ground": _ground_view(),
    }


def _serialize_scorecard():
    g = GAME
    innings = list(g["completed_innings"])
    if g["phase"] != "finished" and g["state"] is not None:
        st = g["state"]
        innings.append({
            "batting_team": g["batting_side"],
            "batting_team_name": g["teams"][g["batting_side"]]["name"],
            "bowling_team_name": g["teams"][_bowling_side()]["name"],
            "batting_team_color": g["teams"][g["batting_side"]].get("color"),
            "bowling_team_color": g["teams"][_bowling_side()].get("color"),
            "runs": st.runs, "wickets": st.wickets,
            "overs": _overs_str(st.balls), "extras": st.extras,
            "batting": list(g["bat_card"].values()),
            "bowling": list(g["bowl_card"].values()),
            "in_progress": True,
        })
    return innings


# --- API ---------------------------------------------------------------------

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    return jsonify({"status": "error", "message": str(e),
                    "trace": traceback.format_exc()}), 500


def _apply_color(role, data):
    c = data.get("color")
    if c in TEAM_COLOR_PALETTE:
        GAME["teams"][role]["color"] = c


@app.route("/api/create_game", methods=["POST"])
def create_game():
    global GAME
    with LOCK:
        GAME = _register_game(_fresh_game())
        data = request.get_json(silent=True) or {}
        token = secrets.token_hex(8)
        GAME["tokens"][token] = "team1"
        GAME["teams"]["team1"]["joined"] = True
        GAME["teams"]["team1"]["name"] = data.get("name") or "Team 1"
        _apply_color("team1", data)
        _register_token(GAME["code"], token)
        _bump()
        return jsonify({"status": "success", "token": token, "code": GAME["code"], "role": "team1"})


@app.route("/api/join_game", methods=["POST"])
def join_game():
    global GAME
    with LOCK:
        data = request.get_json(silent=True) or {}
        code = (data.get("code") or "").strip().upper()
        GAME = _game_by_code(code)
        if GAME is None:
            return jsonify({"status": "error", "message": "No game found with that code."}), 404
        if GAME["teams"]["team2"]["joined"]:
            if data.get("rejoin"):
                token = secrets.token_hex(8)
                GAME["tokens"][token] = "team2"
                _register_token(GAME["code"], token)
                _bump()
                return jsonify({"status": "success", "token": token, "code": GAME["code"], "role": "team2"})
            return jsonify({"status": "error", "message": "This game is already full.", "full": True}), 400
        token = secrets.token_hex(8)
        GAME["tokens"][token] = "team2"
        GAME["teams"]["team2"]["joined"] = True
        GAME["teams"]["team2"]["name"] = data.get("name") or "Team 2"
        _apply_color("team2", data)
        _register_token(GAME["code"], token)
        _bump()
        return jsonify({"status": "success", "token": token, "code": GAME["code"], "role": "team2"})


TOURNAMENT_MIN, TOURNAMENT_MAX = 3, 8

@app.route("/api/create_tournament", methods=["POST"])
def create_tournament():
    """Create a tournament for 3-8 teams. The creator takes the first slot;
    everyone else joins with the code (see /api/join_tournament)."""
    global GAME
    with LOCK:
        data = request.get_json(silent=True) or {}
        try:
            size = int(data.get("size", 4))
        except (TypeError, ValueError):
            size = 4
        if not (TOURNAMENT_MIN <= size <= TOURNAMENT_MAX):
            return jsonify({"status": "error",
                            "message": f"Tournament size must be {TOURNAMENT_MIN}-{TOURNAMENT_MAX} teams."}), 400
        GAME = _register_game(_fresh_game(num_teams=size))
        GAME["tournament"] = {"size": size}
        token = secrets.token_hex(8)
        role = GAME["team_ids"][0]
        GAME["tokens"][token] = role
        GAME["teams"][role]["joined"] = True
        GAME["teams"][role]["name"] = data.get("name") or GAME["teams"][role]["name"]
        _apply_color(role, data)
        _register_token(GAME["code"], token)
        _bump()
        return jsonify({"status": "success", "token": token, "code": GAME["code"], "role": role})


@app.route("/api/join_tournament", methods=["POST"])
def join_tournament():
    """Join an empty slot, OR — if the tournament is full and you already had
    a slot but lost your session (closed the tab, phone locked, cleared
    storage, accidentally clicked something destructive) — reclaim your team
    by passing `rejoin_team_id`. Rejoining just mints a fresh token for that
    slot; it never re-checks who you were, since this is a casual game among
    friends who already have the private room code."""
    global GAME
    with LOCK:
        data = request.get_json(silent=True) or {}
        code = (data.get("code") or "").strip().upper()
        GAME = _game_by_code(code)
        if GAME is None or not GAME.get("tournament"):
            return jsonify({"status": "error", "message": "No tournament found with that code."}), 404

        rejoin_team = data.get("rejoin_team_id")
        if rejoin_team:
            if rejoin_team not in GAME["team_ids"] or not GAME["teams"][rejoin_team]["joined"]:
                return jsonify({"status": "error", "message": "That team isn't available to rejoin."}), 400
            token = secrets.token_hex(8)
            GAME["tokens"][token] = rejoin_team
            _register_token(GAME["code"], token)
            _bump()
            return jsonify({"status": "success", "token": token, "code": GAME["code"], "role": rejoin_team})

        role = next((t for t in GAME["team_ids"] if not GAME["teams"][t]["joined"]), None)
        if role is None:
            roster = [{"team_id": t, "name": GAME["teams"][t]["name"]} for t in GAME["team_ids"]]
            return jsonify({"status": "error", "message": "This tournament is already full.",
                            "full": True, "roster": roster}), 400
        token = secrets.token_hex(8)
        GAME["tokens"][token] = role
        GAME["teams"][role]["joined"] = True
        GAME["teams"][role]["name"] = data.get("name") or GAME["teams"][role]["name"]
        _apply_color(role, data)
        _register_token(GAME["code"], token)
        _bump()
        return jsonify({"status": "success", "token": token, "code": GAME["code"], "role": role})


@app.route("/api/tournament_ready", methods=["POST"])
def tournament_ready():
    """Confirm you're ready for the fixture the ready-gate is currently
    holding on. Once both of that fixture's teams confirm, it starts (toss)."""
    global GAME
    with LOCK:
        token = (request.get_json(silent=True) or {}).get("token", "")
        GAME = _game_by_token(token)
        if GAME is None:
            return jsonify({"status": "error", "message": "Game not found or session expired."}), 404
        role = _role_of(token)
        if not role or not GAME.get("tournament"):
            return jsonify({"status": "error", "message": "Not in a tournament."}), 400
        try:
            _confirm_tournament_ready(role)
        except ValueError as e:
            return jsonify({"status": "error", "message": str(e)}), 400
        _bump()
        return jsonify({"status": "success"})


@app.route("/api/exit_game", methods=["POST"])
def exit_game():
    """Forfeit. Plain 1v1 game: the caller leaves, the opponent wins, the whole
    game ends. Tournament: only the caller's CURRENT fixture is forfeited (and
    they're eliminated from the rest of the tournament) — everyone else keeps
    playing (see _eliminate_from_tournament)."""
    global GAME
    with LOCK:
        token = (request.get_json(silent=True) or {}).get("token", "")
        GAME = _game_by_token(token)
        if GAME is None:
            return jsonify({"status": "success"})
        role = _role_of(token)
        if role is None:
            return jsonify({"status": "success"})
        if GAME.get("tournament"):
            _eliminate_from_tournament(role)
        else:
            match_teams = GAME.get("match_teams") or GAME["team_ids"][:2]
            opp = next((t for t in match_teams if t != role), match_teams[0])
            GAME["abandoned"] = True
            GAME["abandoned_by"] = role
            GAME["phase"] = "finished"
            GAME["result"] = f"{GAME['teams'][opp]['name']} win — {GAME['teams'][role]['name']} left the match."
        _bump()
        return jsonify({"status": "success"})


@app.route("/api/cancel_tournament", methods=["POST"])
def cancel_tournament():
    """Any team can call this to end the WHOLE tournament outright for
    everyone -- unlike exit_game (which only forfeits the caller's own
    fixture and lets everyone else keep playing), this stops it entirely.
    Covers the case where a tournament is stuck (e.g. someone vanished
    before any fixture even existed, a known gap in _eliminate_from_tournament)
    or the group just wants to call it off."""
    global GAME
    with LOCK:
        token = (request.get_json(silent=True) or {}).get("token", "")
        GAME = _game_by_token(token)
        if GAME is None:
            return jsonify({"status": "success"})
        role = _role_of(token)
        t = GAME.get("tournament")
        if role is None or not t:
            return jsonify({"status": "error", "message": "Not in a tournament."}), 400
        if GAME.get("abandoned") or t.get("stage") == "champion":
            return jsonify({"status": "error", "message": "Tournament already over."}), 400
        GAME["abandoned"] = True
        GAME["abandoned_by"] = "__draw__"
        GAME["phase"] = "finished"
        GAME["result"] = f"Tournament cancelled by {GAME['teams'][role]['name']}."
        _bump()
        return jsonify({"status": "success"})


@app.route("/api/state")
def api_state():
    global GAME
    token = request.args.get("token", "")
    with LOCK:
        GAME = _game_by_token(token)
        if GAME is None:
            return jsonify({"status": "no_game"})
        GAME["last_seen"] = time.time()   # polling counts as activity too
        return jsonify(_serialize(token))


@app.route("/api/fixture_scorecard")
def fixture_scorecard():
    """On-demand scorecard for a COMPLETED tournament fixture -- not part of
    the regular polled state (would bloat every /api/state response for
    every player), fetched only when someone actually clicks a played
    fixture in the bracket."""
    global GAME
    token = request.args.get("token", "")
    with LOCK:
        GAME = _game_by_token(token)
        if GAME is None:
            return jsonify({"status": "error", "message": "Game not found or session expired."}), 404
        role = _role_of(token)
        if role is None:
            return jsonify({"status": "error", "message": "Unknown player."}), 403
        t = GAME.get("tournament")
        if not t:
            return jsonify({"status": "error", "message": "Not a tournament."}), 400
        try:
            idx = int(request.args.get("fixture_idx", ""))
        except ValueError:
            return jsonify({"status": "error", "message": "Invalid fixture."}), 400
        fixtures = t.get("fixtures", [])
        if idx < 0 or idx >= len(fixtures) or not fixtures[idx]["played"]:
            return jsonify({"status": "error", "message": "That fixture hasn't been played."}), 400
        innings = (t.get("fixture_scorecards") or {}).get(idx)
        if innings is None:
            return jsonify({"status": "error", "message": "No scorecard available for that fixture (walkover)."}), 404
        fx = fixtures[idx]
        return jsonify({
            "status": "success", "innings": innings,
            "a_name": GAME["teams"][fx["a"]]["name"], "b_name": GAME["teams"][fx["b"]]["name"],
            "kind": fx["kind"], "result_text": fx["result_text"], "motm_name": fx.get("motm"),
        })


@app.route("/api/quick_match", methods=["POST"])
def quick_match():
    global GAME
    with LOCK:
        token = (request.get_json(silent=True) or {}).get("token", "")
        GAME = _game_by_token(token)
        if GAME is None:
            return jsonify({"status": "error", "message": "Game not found or session expired."}), 404
        _require_both_joined()
        t1_xi, t2_xi, t1_roster, t2_roster = _auto_two_xis()
        GAME["teams"]["team1"]["xi"] = t1_xi
        GAME["teams"]["team2"]["xi"] = t2_xi
        # populate squads (XI + 2 reserves each) so Impact Player has a bench
        # to draw from -- Quick Match otherwise never touches GAME["squads"]
        GAME["squads"] = {
            t: {**_new_squad(), "roster": roster, "locked": True,
                "os": sum(1 for p in roster if p.get("is_foreigner")),
                "wk": sum(1 for p in roster if p.get("is_keeper"))}
            for t, roster in (("team1", t1_roster), ("team2", t2_roster))
        }
        GAME["match_teams"] = GAME["team_ids"][:2]
        GAME["match_ground"] = random.choice(STADIUMS)
        _start_single_match()
        _bump()
        return jsonify({"status": "success"})


# --- Auction endpoints -------------------------------------------------------

@app.route("/api/start_auction", methods=["POST"])
def start_auction():
    """Both teams must agree before the auction begins."""
    global GAME
    with LOCK:
        token = (request.get_json(silent=True) or {}).get("token", "")
        GAME = _game_by_token(token)
        if GAME is None:
            return jsonify({"status": "error", "message": "Game not found or session expired."}), 404
        _require_both_joined()
        role = _role_of(token)
        if role is None:
            return jsonify({"status": "error", "message": "Unknown player."}), 403
        if GAME["phase"] != "lobby":
            return jsonify({"status": "error", "message": "Already started."}), 400
        GAME["start_votes"][role] = True
        if all(GAME["start_votes"][t] for t in GAME["team_ids"]):
            _start_auction()
        _bump()
        return jsonify({"status": "success"})


@app.route("/api/auction_ready", methods=["POST"])
def auction_ready():
    """Ready gate: used both for the pool preview and between every lot."""
    global GAME
    with LOCK:
        token = (request.get_json(silent=True) or {}).get("token", "")
        GAME = _game_by_token(token)
        if GAME is None:
            return jsonify({"status": "error", "message": "Game not found or session expired."}), 404
        role = _role_of(token)
        a = GAME["auction"]
        if not role:
            return jsonify({"status": "error", "message": "Unknown player."}), 403
        if a["stage"] == "preview":
            a["ready"][role] = True
            if all(a["ready"][t] for t in GAME["team_ids"]):
                a["set_index"] = 0
                a["player_index"] = 0
                _present_player()
            _bump()
            return jsonify({"status": "success"})
        if a["stage"] == "resolved":
            a["ready"][role] = True
            _resolved_advance()
            _bump()
            return jsonify({"status": "success"})
        return jsonify({"status": "error", "message": "Nothing to ready up for."}), 400


@app.route("/api/bid", methods=["POST"])
def bid():
    global GAME
    with LOCK:
        data = request.get_json(silent=True) or {}
        GAME = _game_by_token(data.get("token", ""))
        if GAME is None:
            return jsonify({"status": "error", "message": "Game not found or session expired."}), 404
        role = _role_of(data.get("token", ""))
        if not role:
            return jsonify({"status": "error", "message": "Unknown player."}), 403
        try:
            _place_bid(role, data.get("amount", 0))
        except ValueError as e:
            return jsonify({"status": "error", "message": str(e)}), 400
        _bump()
        return jsonify({"status": "success"})


@app.route("/api/pull_out", methods=["POST"])
def pull_out():
    global GAME
    with LOCK:
        token = (request.get_json(silent=True) or {}).get("token", "")
        GAME = _game_by_token(token)
        if GAME is None:
            return jsonify({"status": "error", "message": "Game not found or session expired."}), 404
        role = _role_of(token)
        if not role:
            return jsonify({"status": "error", "message": "Unknown player."}), 403
        try:
            _pull_out(role)
        except ValueError as e:
            return jsonify({"status": "error", "message": str(e)}), 400
        _bump()
        return jsonify({"status": "success"})


@app.route("/api/vote_skip_set", methods=["POST"])
def vote_skip_set():
    """Vote to skip whatever's left of the CURRENT set. Once every active
    (unlocked) team has voted yes, the auction jumps straight to the next
    set -- a one-way vote, like auction_ready, not a toggle."""
    global GAME
    with LOCK:
        token = (request.get_json(silent=True) or {}).get("token", "")
        GAME = _game_by_token(token)
        if GAME is None:
            return jsonify({"status": "error", "message": "Game not found or session expired."}), 404
        role = _role_of(token)
        if not role or GAME.get("phase") != "auction":
            return jsonify({"status": "error", "message": "Not in the auction."}), 400
        a = GAME["auction"]
        if a["stage"] not in ("bidding", "resolved"):
            return jsonify({"status": "error", "message": "Can't vote to skip right now."}), 400
        if GAME["squads"][role]["locked"]:
            return jsonify({"status": "error", "message": "Your squad is locked."}), 400
        if _skip_would_strand_squads():
            return jsonify({"status": "error",
                             "message": "Can't skip — too few players would be left for every team to build a full squad."}), 400
        a.setdefault("skip_votes", {})[role] = True
        active = [t for t in GAME["team_ids"] if not GAME["squads"][t]["locked"]]
        if active and all(a["skip_votes"].get(t) for t in active):
            _skip_to_next_set()
        _bump()
        return jsonify({"status": "success"})


@app.route("/api/lock_squad", methods=["POST"])
def lock_squad():
    """Lock in the squad when it satisfies the auction rules (15-21 players,
    >=1 keeper; NO overseas limit here). When both teams lock, go to XI. A
    locked team auto-passes any remaining lots."""
    global GAME
    with LOCK:
        token = (request.get_json(silent=True) or {}).get("token", "")
        GAME = _game_by_token(token)
        if GAME is None:
            return jsonify({"status": "error", "message": "Game not found or session expired."}), 404
        role = _role_of(token)
        if not role or GAME.get("phase") != "auction":
            return jsonify({"status": "error", "message": "Not in an auction."}), 400
        sq = GAME["squads"][role]
        n = len(sq["roster"])
        if n < SQUAD_MIN or n > SQUAD_MAX:
            return jsonify({"status": "error", "message": f"Squad must be {SQUAD_MIN}-{SQUAD_MAX} players (have {n})."}), 400
        if sq["wk"] < 1:
            return jsonify({"status": "error", "message": "You need at least 1 wicket-keeper."}), 400
        sq["locked"] = True
        a = GAME["auction"]
        if all(GAME["squads"][t]["locked"] for t in GAME["team_ids"]):
            _to_grounds()
        elif a["stage"] == "resolved":
            _resolved_advance()      # locked side counts as ready
        elif a["stage"] == "bidding":
            a["out"][role] = True    # drop out of the live lot
        _bump()
        return jsonify({"status": "success"})


@app.route("/api/claim_ground", methods=["POST"])
def claim_ground():
    """Claim (or switch to) a home ground during the grounds phase. Uniqueness
    is enforced here: a stadium anyone else currently holds is rejected, so
    conflicting teams have to coordinate and pick different ones."""
    global GAME
    with LOCK:
        data = request.get_json(silent=True) or {}
        GAME = _game_by_token(data.get("token", ""))
        if GAME is None:
            return jsonify({"status": "error", "message": "Game not found or session expired."}), 404
        role = _role_of(data.get("token", ""))
        if not role or GAME.get("phase") != "grounds":
            return jsonify({"status": "error", "message": "Not selecting grounds."}), 400
        gp = GAME["ground_pick"]
        if gp[role]["locked"]:
            return jsonify({"status": "error", "message": "You already locked your home ground."}), 400
        gid = data.get("ground_id")
        if gid not in STADIUM_BY_ID:
            return jsonify({"status": "error", "message": "Unknown stadium."}), 400
        holder = next((t for t in GAME["team_ids"] if t != role and gp[t]["ground"] == gid), None)
        if holder:
            return jsonify({"status": "error",
                            "message": f"{GAME['teams'][holder]['name']} has already claimed {STADIUM_BY_ID[gid]['name']} — pick another."}), 400
        gp[role]["ground"] = gid
        _bump()
        return jsonify({"status": "success"})


@app.route("/api/lock_ground", methods=["POST"])
def lock_ground():
    global GAME
    with LOCK:
        token = (request.get_json(silent=True) or {}).get("token", "")
        GAME = _game_by_token(token)
        if GAME is None:
            return jsonify({"status": "error", "message": "Game not found or session expired."}), 404
        role = _role_of(token)
        if not role or GAME.get("phase") != "grounds":
            return jsonify({"status": "error", "message": "Not selecting grounds."}), 400
        gp = GAME["ground_pick"]
        if gp[role]["ground"] is None:
            return jsonify({"status": "error", "message": "Claim a stadium first."}), 400
        gp[role]["locked"] = True
        if all(gp[t]["locked"] for t in GAME["team_ids"]):
            if GAME.get("tournament"):
                # Tournaments pick a fresh XI before EVERY fixture anyway (see
                # _set_awaiting_xi) -- asking again right after the auction,
                # before fixture 1 even exists, was a pointless extra step.
                GAME["phase"] = "match"
                _start_tournament_matches()
            else:
                _to_xi()
        _bump()
        return jsonify({"status": "success"})


@app.route("/api/toggle_xi", methods=["POST"])
def toggle_xi():
    global GAME
    with LOCK:
        data = request.get_json(silent=True) or {}
        GAME = _game_by_token(data.get("token", ""))
        if GAME is None:
            return jsonify({"status": "error", "message": "Game not found or session expired."}), 404
        role = _role_of(data.get("token", ""))
        if not role or GAME["phase"] != "xi":
            return jsonify({"status": "error", "message": "Not selecting XI."}), 400
        sel = GAME["xi_select"][role]
        if sel["locked"]:
            return jsonify({"status": "error", "message": "XI already locked."}), 400
        name = data.get("player_name")
        if name not in [p["name"] for p in GAME["squads"][role]["roster"]]:
            return jsonify({"status": "error", "message": "Player not in your squad."}), 400
        if name in sel["xi"]:
            sel["xi"].remove(name)
        elif len(sel["xi"]) < XI_SIZE:
            sel["xi"].append(name)
        _bump()
        return jsonify({"status": "success"})


@app.route("/api/lock_xi", methods=["POST"])
def lock_xi():
    global GAME
    with LOCK:
        token = (request.get_json(silent=True) or {}).get("token", "")
        GAME = _game_by_token(token)
        if GAME is None:
            return jsonify({"status": "error", "message": "Game not found or session expired."}), 404
        role = _role_of(token)
        if not role or GAME["phase"] != "xi":
            return jsonify({"status": "error", "message": "Not selecting XI."}), 400
        sel = GAME["xi_select"][role]
        roster = {p["name"]: p for p in GAME["squads"][role]["roster"]}
        if len(sel["xi"]) != XI_SIZE:
            return jsonify({"status": "error", "message": f"Pick exactly {XI_SIZE} players."}), 400
        os_in = sum(1 for n in sel["xi"] if roster[n].get("is_foreigner"))
        wk_in = sum(1 for n in sel["xi"] if roster[n].get("is_keeper") or roster[n].get("assigned_role") == "Wicket Keeper")
        if os_in > XI_MAX_OVERSEAS:
            return jsonify({"status": "error", "message": f"Max {XI_MAX_OVERSEAS} overseas players."}), 400
        if wk_in < 1:
            return jsonify({"status": "error", "message": "Need at least 1 wicket-keeper in your XI."}), 400
        sel["locked"] = True
        if all(GAME["xi_select"][t]["locked"] for t in GAME["team_ids"]):
            _finalize_xi_to_match()
        _bump()
        return jsonify({"status": "success"})


@app.route("/api/reorder_xi", methods=["POST"])
def reorder_xi():
    """Drag-and-drop reordering of the chosen XI (cosmetic/organizational --
    the engine still picks openers/next-batter freely by name during play)."""
    global GAME
    with LOCK:
        data = request.get_json(silent=True) or {}
        GAME = _game_by_token(data.get("token", ""))
        if GAME is None:
            return jsonify({"status": "error", "message": "Game not found or session expired."}), 404
        role = _role_of(data.get("token", ""))
        if not role or GAME["phase"] != "xi":
            return jsonify({"status": "error", "message": "Not selecting XI."}), 400
        sel = GAME["xi_select"][role]
        if sel["locked"]:
            return jsonify({"status": "error", "message": "XI already locked."}), 400
        order = data.get("order")
        if not isinstance(order, list) or set(order) != set(sel["xi"]):
            return jsonify({"status": "error", "message": "Invalid order."}), 400
        sel["xi"] = order
        _bump()
        return jsonify({"status": "success"})


@app.route("/api/toggle_fixture_xi", methods=["POST"])
def toggle_fixture_xi():
    """Per-tournament-fixture XI pick/unpick (a fresh selection before every
    match, not just once for the whole tournament -- see _set_awaiting_xi)."""
    global GAME
    with LOCK:
        data = request.get_json(silent=True) or {}
        GAME = _game_by_token(data.get("token", ""))
        if GAME is None:
            return jsonify({"status": "error", "message": "Game not found or session expired."}), 404
        role = _role_of(data.get("token", ""))
        t = GAME.get("tournament") or {}
        if not role or not t.get("awaiting_xi") or role not in t.get("fixture_xi", {}):
            return jsonify({"status": "error", "message": "Not selecting an XI right now."}), 400
        sel = t["fixture_xi"][role]
        if sel["locked"]:
            return jsonify({"status": "error", "message": "XI already locked."}), 400
        name = data.get("player_name")
        if name not in [p["name"] for p in GAME["squads"][role]["roster"]]:
            return jsonify({"status": "error", "message": "Player not in your squad."}), 400
        if name in sel["xi"]:
            sel["xi"].remove(name)
        elif len(sel["xi"]) < XI_SIZE:
            sel["xi"].append(name)
        _bump()
        return jsonify({"status": "success"})


@app.route("/api/reorder_fixture_xi", methods=["POST"])
def reorder_fixture_xi():
    global GAME
    with LOCK:
        data = request.get_json(silent=True) or {}
        GAME = _game_by_token(data.get("token", ""))
        if GAME is None:
            return jsonify({"status": "error", "message": "Game not found or session expired."}), 404
        role = _role_of(data.get("token", ""))
        t = GAME.get("tournament") or {}
        if not role or not t.get("awaiting_xi") or role not in t.get("fixture_xi", {}):
            return jsonify({"status": "error", "message": "Not selecting an XI right now."}), 400
        sel = t["fixture_xi"][role]
        if sel["locked"]:
            return jsonify({"status": "error", "message": "XI already locked."}), 400
        order = data.get("order")
        if not isinstance(order, list) or set(order) != set(sel["xi"]):
            return jsonify({"status": "error", "message": "Invalid order."}), 400
        sel["xi"] = order
        _bump()
        return jsonify({"status": "success"})


@app.route("/api/lock_fixture_xi", methods=["POST"])
def lock_fixture_xi():
    global GAME
    with LOCK:
        token = (request.get_json(silent=True) or {}).get("token", "")
        GAME = _game_by_token(token)
        if GAME is None:
            return jsonify({"status": "error", "message": "Game not found or session expired."}), 404
        role = _role_of(token)
        if not role:
            return jsonify({"status": "error", "message": "Unknown player."}), 403
        try:
            _lock_fixture_xi(role)
        except ValueError as e:
            return jsonify({"status": "error", "message": str(e)}), 400
        _bump()
        return jsonify({"status": "success"})


@app.route("/api/toss_choice", methods=["POST"])
def toss_choice():
    global GAME
    with LOCK:
        data = request.get_json(silent=True) or {}
        GAME = _game_by_token(data.get("token", ""))
        if GAME is None:
            return jsonify({"status": "error", "message": "Game not found or session expired."}), 404
        role = _role_of(data.get("token", ""))
        if GAME["stage"] != "toss":
            return jsonify({"status": "error", "message": "Toss already done."}), 400
        if role != GAME["toss"]["winner"]:
            return jsonify({"status": "error", "message": "You did not win the toss."}), 403
        choice = data.get("choice")
        if choice not in ("bat", "bowl"):
            return jsonify({"status": "error", "message": "Choose bat or bowl."}), 400
        opp = next(t for t in GAME["match_teams"] if t != role)
        batting_side = role if choice == "bat" else opp
        GAME["toss"]["decided"] = True
        GAME["toss"]["choice"] = choice
        GAME["innings"] = 1
        _prepare_innings(batting_side, target=None)
        _bump()
        return jsonify({"status": "success"})


@app.route("/api/set_openers", methods=["POST"])
def set_openers():
    global GAME
    with LOCK:
        data = request.get_json(silent=True) or {}
        GAME = _game_by_token(data.get("token", ""))
        if GAME is None:
            return jsonify({"status": "error", "message": "Game not found or session expired."}), 404
        role = _role_of(data.get("token", ""))
        if role != GAME["batting_side"]:
            return jsonify({"status": "error", "message": "Only the batting side picks openers."}), 403
        if GAME["stage"] != "openers":
            return jsonify({"status": "error", "message": "Openers already set."}), 400
        s_name, ns_name = data.get("striker"), data.get("non_striker")
        st = GAME["state"]
        names = [b.name for b in st.lineup]
        if s_name not in names or ns_name not in names or s_name == ns_name:
            return jsonify({"status": "error", "message": "Pick two different batsmen from your XI."}), 400
        st.striker_index = names.index(s_name)
        st.non_striker_index = names.index(ns_name)
        GAME["used_batters"] = [s_name, ns_name]
        _ensure_bat_row(s_name)
        _ensure_bat_row(ns_name)
        GAME["stage"] = "play"
        _bump()
        return jsonify({"status": "success"})


@app.route("/api/ready_resume", methods=["POST"])
def ready_resume():
    """Both sides confirm (and may re-set intent) after a new batsman walks in
    mid-over, before the over resumes -- the bowling side gets a say too,
    since a fresh batter at the crease can be worth attacking or containing
    differently than whoever was just dismissed."""
    global GAME
    with LOCK:
        data = request.get_json(silent=True) or {}
        GAME = _game_by_token(data.get("token", ""))
        if GAME is None:
            return jsonify({"status": "error", "message": "Game not found or session expired."}), 404
        role = _role_of(data.get("token", ""))
        if GAME["stage"] != "await_resume":
            return jsonify({"status": "error", "message": "Nothing to resume."}), 400
        ao = GAME["active_over"]
        if role == GAME["batting_side"]:
            ao["striker_intent"] = int(data.get("striker_intent", ao["striker_intent"]))
            ao["non_striker_intent"] = int(data.get("non_striker_intent", ao["non_striker_intent"]))
            ao["striker_role"] = _clean_bat_role(data.get("striker_role", ao.get("striker_role")))
            ao["non_striker_role"] = _clean_bat_role(data.get("non_striker_role", ao.get("non_striker_role")))
            # register whoever is now at the crease (the new batter, and the
            # survivor) by NAME, so the rest of the over stays identity-locked
            # (intent AND role) even through a further rotation or 2nd wicket
            st = GAME["state"]
            by_name = ao.setdefault("intent_by_name", {})
            by_role = ao.setdefault("role_by_name", {})
            s, ns = st.get_striker(), st.get_non_striker()
            if s:
                by_name[s.name] = ao["striker_intent"]
                by_role[s.name] = ao["striker_role"]
            if ns:
                by_name[ns.name] = ao["non_striker_intent"]
                by_role[ns.name] = ao["non_striker_role"]
            ao["resume_batting_ready"] = True
        elif role == _bowling_side():
            ao["bowl_intent"] = int(data.get("bowl_intent", ao["bowl_intent"]))
            ao["bowl_role"] = _clean_bowl_role(data.get("bowl_role", ao.get("bowl_role")))
            ao["resume_bowling_ready"] = True
        else:
            return jsonify({"status": "error", "message": "Unknown player."}), 403
        if ao.get("resume_batting_ready") and ao.get("resume_bowling_ready"):
            GAME["stage"] = "play"
            _run_and_route()
        else:
            _bump()
        return jsonify({"status": "success"})


@app.route("/api/free_hit", methods=["POST"])
def free_hit():
    """Both sides may adjust intent for the single free-hit delivery, then ready."""
    global GAME
    with LOCK:
        data = request.get_json(silent=True) or {}
        GAME = _game_by_token(data.get("token", ""))
        if GAME is None:
            return jsonify({"status": "error", "message": "Game not found or session expired."}), 404
        role = _role_of(data.get("token", ""))
        if GAME["stage"] != "free_hit":
            return jsonify({"status": "error", "message": "No free hit pending."}), 400
        fh = GAME["free_hit"]
        if role == GAME["batting_side"]:
            fh["striker_intent"] = int(data.get("striker_intent", fh["striker_intent"]))
            fh["non_striker_intent"] = int(data.get("non_striker_intent", fh["non_striker_intent"]))
            fh["striker_role"] = _clean_bat_role(data.get("striker_role", fh.get("striker_role")))
            fh["non_striker_role"] = _clean_bat_role(data.get("non_striker_role", fh.get("non_striker_role")))
            fh["batting_ready"] = True
        else:
            fh["bowl_intent"] = int(data.get("bowl_intent", fh["bowl_intent"]))
            fh["bowl_role"] = _clean_bowl_role(data.get("bowl_role", fh.get("bowl_role")))
            fh["bowling_ready"] = True
        if fh["batting_ready"] and fh["bowling_ready"]:
            fh["active"] = False
            GAME["stage"] = "play"
            _run_and_route()
        else:
            _bump()
        return jsonify({"status": "success"})


@app.route("/api/impact_sub", methods=["POST"])
def impact_sub():
    """One bench-for-XI swap per team, once for the whole match -- see
    _apply_impact_sub for the exact checkpoints/eligibility rules."""
    global GAME
    with LOCK:
        data = request.get_json(silent=True) or {}
        token = data.get("token", "")
        GAME = _game_by_token(token)
        if GAME is None:
            return jsonify({"status": "error", "message": "Game not found or session expired."}), 404
        role = _role_of(token)
        if role is None or GAME.get("phase") != "match":
            return jsonify({"status": "error", "message": "Not in a live match."}), 400
        out_name = data.get("out_name")
        in_name = data.get("in_name")
        if not out_name or not in_name:
            return jsonify({"status": "error", "message": "Pick both an outgoing and incoming player."}), 400
        try:
            _apply_impact_sub(role, out_name, in_name)
        except ValueError as e:
            return jsonify({"status": "error", "message": str(e)}), 400
        _bump()
        return jsonify({"status": "success"})


@app.route("/api/retire_batsman", methods=["POST"])
def retire_batsman():
    """Permanently retire the striker or non-striker. Only between overs (stage
    'play', before either half of the next over has been submitted) so the
    ball-by-ball over loop never needs to handle a mid-over interrupt."""
    global GAME
    with LOCK:
        data = request.get_json(silent=True) or {}
        GAME = _game_by_token(data.get("token", ""))
        if GAME is None:
            return jsonify({"status": "error", "message": "Game not found or session expired."}), 404
        role = _role_of(data.get("token", ""))
        if role != GAME.get("batting_side"):
            return jsonify({"status": "error", "message": "Only the batting side can retire a batter."}), 403
        if GAME["phase"] != "match" or GAME["stage"] != "play":
            return jsonify({"status": "error", "message": "Can only retire a batter between overs."}), 400
        if GAME["pending_over"]["batting"]["submitted"] or GAME["pending_over"]["bowling"]["submitted"]:
            return jsonify({"status": "error", "message": "Too late — the next over is already being set up."}), 400

        which = data.get("which")
        if which not in ("striker", "non_striker"):
            return jsonify({"status": "error", "message": "which must be 'striker' or 'non_striker'."}), 400

        st = GAME["state"]
        batter = st.get_striker() if which == "striker" else st.get_non_striker()
        if batter is None:
            return jsonify({"status": "error", "message": "No batter there to retire."}), 400

        row = GAME["bat_card"][batter.name]
        row["out"] = True
        row["how_out"] = "retired hurt"
        st.retire(which)
        GAME["vacant_slot"] = which

        if st.is_all_out():
            _handle_innings_over()
        else:
            GAME["stage"] = "await_batter"
        _bump()
        return jsonify({"status": "success"})


@app.route("/api/submit_over", methods=["POST"])
def submit_over():
    global GAME
    with LOCK:
        data = request.get_json(silent=True) or {}
        token = data.get("token", "")
        GAME = _game_by_token(token)
        if GAME is None:
            return jsonify({"status": "error", "message": "Game not found or session expired."}), 404
        role = _role_of(token)
        if role is None:
            return jsonify({"status": "error", "message": "Unknown player."}), 403
        if GAME["phase"] != "match" or GAME["stage"] != "play":
            return jsonify({"status": "error", "message": "Not ready for a new over yet."}), 400

        i_bat = role == GAME["batting_side"]
        wants_gambit = bool(data.get("gambit"))
        cards = (GAME.get("gambit_cards") or {}).get(role, {})
        if wants_gambit and not cards.get("attack" if i_bat else "trap"):
            return jsonify({"status": "error", "message": "Gambit already used this match."}), 400
        if i_bat:
            # sequenced over: the bowler must be locked in first
            if not GAME["pending_over"]["bowling"]["submitted"]:
                return jsonify({"status": "error",
                                "message": "Wait for the bowling side to lock in the bowler."}), 400
            GAME["pending_over"]["batting"] = {
                "submitted": True,
                "striker_intent": int(data.get("striker_intent", 50)),
                "non_striker_intent": int(data.get("non_striker_intent", 50)),
                "striker_role": _clean_bat_role(data.get("striker_role")),
                "non_striker_role": _clean_bat_role(data.get("non_striker_role")),
                "gambit": wants_gambit,
            }
        else:
            bowler_name = data.get("bowler_name")
            if bowler_name not in [p["name"] for p in GAME["teams"][_bowling_side()]["xi"]]:
                return jsonify({"status": "error", "message": "Invalid bowler."}), 400
            if GAME["bowler_stats"].get(bowler_name, 0) >= MAX_OVERS_PER_BOWLER:
                return jsonify({"status": "error", "message": "Bowler has bowled max overs."}), 400
            if bowler_name == GAME["last_bowler"]:
                return jsonify({"status": "error", "message": "Bowler cannot bowl consecutive overs."}), 400
            GAME["pending_over"]["bowling"] = {
                "submitted": True,
                "bowler_name": bowler_name,
                "bowl_intent": int(data.get("bowl_intent", 50)),
                "bowl_role": _clean_bowl_role(data.get("bowl_role")),
                "gambit": wants_gambit,
            }
        GAME["teams"][role]["ready"] = True
        _try_resolve_over()
        _bump()
        return jsonify({"status": "success"})


@app.route("/api/set_next_batter", methods=["POST"])
def set_next_batter():
    global GAME
    with LOCK:
        data = request.get_json(silent=True) or {}
        token = data.get("token", "")
        GAME = _game_by_token(token)
        if GAME is None:
            return jsonify({"status": "error", "message": "Game not found or session expired."}), 404
        role = _role_of(token)
        if role != GAME["batting_side"]:
            return jsonify({"status": "error", "message": "Only the batting side chooses batters."}), 403
        if GAME["stage"] != "await_batter":
            return jsonify({"status": "error", "message": "No batter needed right now."}), 400

        name = data.get("batter_name")
        st = GAME["state"]
        idx = next((i for i, b in enumerate(st.lineup) if b.name == name), None)
        if idx is None:
            return jsonify({"status": "error", "message": "Batter not in XI."}), 400
        if name in GAME["used_batters"]:
            return jsonify({"status": "error", "message": "Batter already used."}), 400

        if GAME.get("vacant_slot") == "non_striker":
            st.non_striker_index = idx
        else:
            st.striker_index = idx
        GAME["used_batters"].append(name)
        _ensure_bat_row(name)

        if GAME["active_over"] is None:
            # no over in progress (a retire between overs, or the very first
            # batter gap of the innings) -> just resume play, nothing to complete
            GAME["stage"] = "play"
            _bump()
        elif st.balls < GAME["over_end_at"]:
            # over still in progress -> BOTH sides must press Ready to resume.
            # Reset both sides' pending-submitted flags so the intent sliders
            # re-enable (otherwise they'd stay disabled/stale from the over's
            # original submission), and clear the resume-ready flags fresh --
            # the bowling side gets a say too, since a new batter at the
            # crease can be worth attacking/containing differently.
            GAME["pending_over"]["batting"]["submitted"] = False
            GAME["pending_over"]["bowling"]["submitted"] = False
            ao = GAME["active_over"]
            ao["resume_batting_ready"] = False
            ao["resume_bowling_ready"] = False
            GAME["stage"] = "await_resume"
            _bump()
        else:
            # wicket fell on the last ball -> the over is done; go to next over
            if st.get_striker() is not None:
                st.rotate_strike()
            _complete_over()
            _bump()
        return jsonify({"status": "success"})


def _require_both_joined():
    if GAME is None or not all(GAME["teams"][t]["joined"] for t in GAME["team_ids"]):
        raise RuntimeError("All teams must join before starting.")


# --- Static assets -----------------------------------------------------------

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)


# Auction heartbeat: advances going-once/twice/sold timers server-side so both
# devices stay in sync. Daemon thread — dies with the process.
threading.Thread(target=_auction_tick, daemon=True).start()


if __name__ == "__main__":
    # Cloud hosts (Railway, etc.) assign a dynamic port via $PORT; falls back
    # to 8000 for local dev (matches run_online.bat and the Cloudflare tunnel).
    port = int(os.environ.get("PORT", 8000))
    print("=" * 45)
    print("CRICKET ATTACK - MULTIPLAYER SERVER")
    print(f"Listening on http://0.0.0.0:{port}")
    print(f"Loaded {len(ALL_PLAYERS)} players | Stage-3 threat_base "
          f"{LEAGUE_AVG['threat_base']:.3f}, patience_base {LEAGUE_AVG['patience_base']:.1f} | "
          f"Stage-2 bat_power_base {LEAGUE_AVG['bat_power_base']:.1f}, "
          f"bowl_power_base {LEAGUE_AVG['bowl_power_base']:.3f}")
    print("=" * 45)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
