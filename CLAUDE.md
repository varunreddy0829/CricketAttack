# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A cricket match simulator built on a deterministic, stage-based probability engine. Real IPL ball-by-ball data (`data/All_Matches_Json/*.json`) is compiled into per-player career stats and OVR ratings, which feed a weighted-random outcome model for simulating overs ball-by-ball. There's a Flask web app for playing a full simulated match, plus a standalone CLI, a draft-pool generator, and a manual data-labeling tool.

`requirements.txt` and `README.md` are currently empty. Dependencies actually used: `flask`. Everything else is stdlib (`json`, `random`, `glob`, `urllib`).

## Commands

Run all commands from the repo root (scripts use repo-relative paths like `data/...`).

- Run tests: `python -m unittest tests/test_simulation.py -v`
- Run a single test: `python -m unittest tests.test_simulation.TestCricketMathEngine.test_stage1_probability_sum`
- CLI simulator (interactive, one over): `python -m src.main`
- Web app (two-device multiplayer, port 8000): `python src/server.py` — open two browsers to `http://<lan-ip>:8000`; one creates a game, the other joins with the code, then Quick Match.
- Player-labeling tool (foreigner tagging UI, port 5007): `python label_server.py`
- Rebuild `data/players_historical.json` from raw match JSON: `python -m src.utils.compile_player_stats`
- Note: `run_mock.py` is stale — it hits the old `/api/init_match`/`/api/play_over` endpoints that the server rewrite removed.

There is no lint/build config in the repo.

## Architecture

### The probability engine pipeline (`src/engine/`)

Every ball is resolved by running a `{'0','1','2','3','4','5','6','Out'}` weight dictionary (always conserved to sum to 1000.0) through sequential stages, each nudging the distribution and re-normalizing. Every batter starts from the exact same global `BASELINE_WEIGHTS` (`stats_calculator.py`) — there is deliberately no per-batter "DNA" stage: an earlier version derived a starting distribution from each batter's own career dismissals/fours/sixes rate before the bowler was even considered, which double-counted the same stats Stage 3 already uses comparatively and distorted `Out` probabilities in both directions (making elite batters look nearly unbeatable and weak batters look far more fragile than the bowler matchup alone warranted).

1. **Stage 1 — OVR ratio** (`apply_stage1_ovr`): scales boundary/run weights by `batter.ovr / bowler.ovr`.
2. **Stage 2 — Batting power vs bowling anti-power** (`apply_stage2_strike_rate_economy`): a secondary (20%-weighted, tunable) nudge from two ghost-stat-smoothed `VOLUME^A × RATE^B` scores — `balls^A × ghost_strike_rate^B` for the batter vs `legal_balls^A × (1/ghost_economy)^B` for the bowler — each normalized against a league-median baseline so an average regular scores 1.0. Exponents/priors/bases live in `config/baseline_weights.json`.
3. **Stage 3 — Wicket factor** (`apply_stage3_wicket_factor`): adjusts `Out` weight from a symmetric pair of ghost-stat-smoothed longevity scores — bowler `wickets^A / ghost_strike_rate^B` ("threat") vs batter `runs^A × ghost_average^B` ("patience", inverted to a defense multiplier) — each normalized against a league-median baseline. Same config file as Stage 2.
4. **Stage 4 — Intent** (`intent_handler.apply_intents`): a user-controlled 0–100 meter per batter/bowler (50 = neutral) that trades off dot-ball probability against run/wicket probability. Applied last, sequentially for striker then bowler.

`simulator.calculate_single_ball` runs stages in this exact order and samples one outcome via `random.choices`. `simulator.simulate_over` wraps this for a 6-legal-ball over: it separately rolls for extras (`EXTRAS_PROB`, outside the weight pipeline), handles strike rotation on odd runs, and stops mid-over on a wicket (the frontend/caller is expected to select the next batter — `MatchState.handle_wicket` deliberately does not auto-select one).

Each stage function conserves the 1000.0 sum independently — when modifying this pipeline, preserve that invariant (tests assert it at every stage) and keep '3' and '5' (rare boundary types) treated as near-static constants rather than model-driven.

### Data flow: raw matches -> ratings -> gameplay

`src/utils/compile_player_stats.py` is the offline ETL: it scans `data/All_Matches_Json/*.json` (Cricsheet-style ball-by-ball JSON), aggregates career batting/bowling totals per player, computes Bayesian-smoothed averages/strike-rates/economies to stabilize small samples, derives a `raw_bat_power`/`raw_bowl_power` score, and scales those onto a 55–99 OVR range relative to the best player in the dataset. Output is written to `data/players_historical.json` (with a `//`-prefixed comment header before the JSON array — callers must `text[text.find('['):]` to strip it before `json.loads`). Foreigner/keeper flags come from hardcoded name sets (`KNOWN_FOREIGNERS`, `KNOWN_KEEPERS`) plus dynamically-detected stumping fielders.

`src/models/player.py` (`Batter`, `Bowler`) is the runtime shape the engine consumes — a thin wrapper that derives `avg`/`sr` from raw career counts at construction time.

### Web app — server-authoritative two-device multiplayer (`src/server.py` + `src/public/`)

The web app is a **networked two-device game**: two browsers connect to one server-held game (one *creates* and gets a 4-char code, the other *joins* with it). **The server is the single source of truth** — all game logic lives in `src/server.py`; the client only renders and posts actions. This is a deliberate consequence of the hidden-intent rule: the batting side must never see the bowler's intent and vice-versa, which is impossible on a shared screen.

- **One global `GAME` dict** (guarded by a `threading.RLock`) with a `phase` state machine: `lobby → auction → xi → match → finished` (or `lobby → match` via **Quick Match**, which auto-drafts two XIs and skips the draft). A `version` int is bumped on every mutation.
- **Within `phase == "match"`, a `stage` sub-machine drives the flow:** `toss` (winner picks bat/bowl via `/api/toss_choice`) → `openers` (batting side picks its opening pair via `/api/set_openers`) → `play` (the over handshake) → on a wicket `await_batter` (`/api/set_next_batter`) → `await_resume` (batting side re-confirms via `/api/ready_resume` before the over continues) → back to `play`. A no-ball diverts to `free_hit`: a one-ball window where **both** sides may re-set intent (`/api/free_hit`) before the free-hit delivery, on which the batter cannot be dismissed. Each stage is enforced server-side; the client (`app.js`) switches its render on `match.stage`.
- **Sync is polling, not websockets.** Clients hold a `player_token` (in `localStorage`), poll `GET /api/state?token=…` ~700ms, and re-render only when `version` changes (`net.js` dispatches a `gamestate` event).
- **Redaction is the security boundary.** `_serialize(token)` builds a per-role view: the opponent's `pending` over submission only ever exposes booleans (`opponent_submitted`), never their intent numbers or bowler pick. `my_bench` is role-specific (your batters, or your bowlers with `overs_bowled`/`disabled`); `opponent_list` is just the other XI's public names/OVRs.
- **The over handshake** (`pending_over`): the batting device POSTs `submit_over` with striker/non-striker intents; the bowling device POSTs its bowler pick + bowl intent. When *both* halves are in, `_try_resolve_over` locks the intents into `active_over` and simulates. Mid-over wickets set `await_next_batter`; the batting side calls `set_next_batter` (drag-and-drop in the UI) and the server auto-resumes the remainder.
- **The server owns the match rules**: the 20-over innings cap (`OVERS_PER_INNINGS`), innings/target/win-loss, the max-4-overs and no-consecutive-overs bowler quota, `used_batters`, and per-batter/per-bowler scorecards.
- **Stage-3 calibration**: `BENCHMARK_CAREER_WICKETS` (150) is the league yardstick fed to the engine's wicket factor. It must track the *career*-wickets scale of players who actually bowl (front-line bowlers have 70–230), NOT the data mean over all 811 players (~27, dragged down by part-timers) — the latter makes every real bowler a 3–8× wicket threat and bowls teams out far too cheaply. Calibrated so balanced-intent innings average ~170 with ~14% all-out.
- **`_simulate_over_rich` / `_simulate_until_pause`** reimplements the over loop *inside `server.py`* so it can capture per-ball, per-batter and per-bowler detail the UI needs — but it still calls the engine's `calculate_single_ball` for the actual probability math (the standalone `simulate_over` in `src/engine/simulator.py` is now only used by the `src/main.py` CLI). It also builds `Batter`/`Bowler` from the authoritative `data/players_historical.json` nested stats (by name), rather than trusting client-passed fields.
- **Quick Match** (`/api/quick_match`, `_auto_two_xis`) auto-drafts two balanced XIs so a match is playable now, before the networked auction is built.

- **Auction phase** (`phase == "auction"`) — server-authoritative IPL-style draft mirroring the original hotseat rules. `generate_draft_pool` (in `engine/draft_generator.py`: 12 sets = 3 tiers × 4 roles × 5 players) is walked lot by lot. Each lot: a `setbreak` (both `/api/auction_ready`) → `bidding` (`/api/bid` raises, `/api/pull_out` folds; ₹100 Cr purse, `BASE_BID`/min-increment 0.5) → `sold`/`unsold` announcement → next. The **going-once/twice/sold timer runs on a background daemon thread** (`_auction_tick`) keyed on `time.time()` deadlines, so both devices see the countdown resolve in sync (this is the one place the server drives state without a client action). After all lots, `done`: `/api/auto_fill` tops a squad up to 15 (keeper-first), `/api/confirm_squad` validates 15–21 with ≥1 keeper. The auction is **public** (bids/purse/rosters visible to both) — only the *match* has hidden intent.
- **XI phase** (`phase == "xi"`) — each device independently picks 11 from its own squad (`/api/toggle_xi`, `/api/lock_xi`): exactly 11, ≤4 overseas, ≥1 keeper. When both lock, `_finalize_xi_to_match` copies each XI into `teams[side].xi` and starts the match (toss). Each side sees only its own squad.

Frontend (`src/public/`): `index.html` (landing / lobby / auction / xi / 3-tab game shell), `style.css` (a "classic scoreboard" theme — field green, cream panels, black-and-amber scoreboard, OVR-tier card borders), `net.js` (polling/token/api), `app.js` (all screens + interactions). The old single-file `main.js` client is gone.

### Labeling tool (`label_server.py`)

Independent single-purpose Flask app (port 5007) for crowdsourcing the `is_foreigner` flag onto `data/players_historical.json` entries, writing to `data/player_labels.json`. Has a simple 120-second soft lock (`active_locks`) per player to avoid two labelers grabbing the same name simultaneously; not safe against real concurrent write races since it isn't used elsewhere in the pipeline yet.

### Config files

`config/baseline_weights.json` and `config/ground_configs.json` are currently empty placeholders — not yet wired into the engine (baseline weights currently live as the `BASELINE_WEIGHTS` constant in `stats_calculator.py`).
