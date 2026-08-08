"""Sequential replay of Cricsheet ball-by-ball JSON into fully-stated ball records.

The raw files carry no running state: no score, no wickets in hand, no balls
remaining, no "how long has this batter been in". All of it is reconstructable by
walking the deliveries in order, which is what this module does.

This is the ONE place eligibility filtering and state derivation live. Both the
real-data reference statistics (ml/harness/reference.py) and the model's feature
table (ml/etl/build_table.py) are built from `iter_innings`, so the two can never
end up measuring differently-filtered populations -- a mismatch there would make
every calibration target silently wrong.

Read-only with respect to everything outside ml/.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

# --- outcome vocabulary ----------------------------------------------------
# 9 classes. '5' is deliberately absent (~1.5 per 10,000 balls, nearly all
# overthrows) and is injected as a constant at simulation time instead of being
# estimated. Wicket *kind* is likewise not a class -- `Out` is ~4.9% and splitting
# it five ways puts classes at 0.2-2.5% estimated against 800+ players, which
# spends real signal to buy flavour text a lookup table gives for free.
CLASSES = ("0", "1", "2", "3", "4", "6", "Out", "wide", "noball")
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}

# Dismissals credited to the bowler-vs-striker matchup. Run outs are handled
# separately (they can befall either batter and aren't the striker's contest),
# and retirements are dropped entirely rather than labelled.
STRIKER_DISMISSALS = frozenset({
    "bowled", "caught", "caught and bowled", "lbw", "stumped",
    "hit wicket", "obstructing the field", "handled the ball",
    "hit the ball twice", "timed out",
})
RETIREMENTS = frozenset({"retired hurt", "retired out", "retired not out"})

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MATCH_GLOB = os.path.join(REPO_ROOT, "data", "All_Matches_Json", "*.json")

DEATH_OVERS = 5  # the death phase is the last N overs of the innings


@dataclass
class BallRecord:
    """One delivery, with all state as it stood BEFORE the ball was bowled."""

    # identity
    match_id: str
    season: int
    venue: str
    innings_no: int          # 1 or 2
    batting_team: str
    bowling_team: str

    # who
    batter: str
    batter_id: str
    bowler: str
    bowler_id: str
    non_striker: str
    non_striker_id: str

    # position in the innings
    over: int                # 0-indexed
    ball_in_over: int        # 1-6, counting legal deliveries only
    legal_index: int         # legal balls bowled before this one
    phase: str               # "pp" | "mid" | "death", from the real powerplay bounds

    # state BEFORE the ball
    score: int
    wickets: int
    balls_remaining: int
    target: int | None       # None in the first innings
    striker_runs: int
    striker_balls: int
    striker_position: int    # 1-11, order of arrival at the crease
    partnership_runs: int
    partnership_balls: int
    bowler_balls: int        # legal balls bowled by him this innings
    bowler_runs: int
    bowler_wickets: int
    over_in_spell: int       # 1 = first over of an unbroken spell
    free_hit: bool

    # outcome
    outcome: str             # one of CLASSES
    runs_batter: int
    runs_total: int
    is_legal: bool
    wicket_kind: str | None  # set only when outcome == "Out"


@dataclass
class InningsRecord:
    """An innings' worth of balls, plus the real lineup and bowling plan.

    The harness replays these verbatim rather than inventing squads -- simulating
    against a synthetic squad sampler measures the sampler, not the engine.
    """

    match_id: str
    season: int
    venue: str
    innings_no: int
    batting_team: str
    bowling_team: str
    target: int | None
    # Batting order by arrival, then the rest of the named XI who never batted.
    # The tail MATTERS: without it a side that lost 6 wickets yields a 7-name
    # lineup, and any simulator replaying it would call 6 down "all out".
    lineup: list[str]
    bowler_by_over: list[str]         # bowler name per over index
    balls: list[BallRecord] = field(default_factory=list)

    # innings result
    total: int = 0
    wickets: int = 0
    legal_balls: int = 0

    @property
    def all_out(self) -> bool:
        return self.wickets >= 10

    @property
    def overs_faced(self) -> float:
        return self.legal_balls / 6.0


# --- eligibility -----------------------------------------------------------

def match_is_eligible(info: dict) -> bool:
    """Standard 20-over IPL matches only.

    Rain-shortened (DLS) games have different powerplay bounds and innings
    lengths, which would pollute the phase and balls-remaining features; they are
    excluded rather than special-cased.
    """
    if info.get("match_type") != "T20":
        return False
    if info.get("overs") != 20:
        return False
    if info.get("balls_per_over", 6) != 6:
        return False
    if (info.get("outcome") or {}).get("method"):   # D/L
        return False
    return True


def _phase_bounds(innings: dict, total_overs: int = 20) -> int:
    """Last over index (0-based) still inside the mandatory powerplay.

    Read from the innings' own `powerplays` block rather than assuming `over <= 5`:
    2,480 of 2,514 innings carry real bounds and a handful deviate (5.7, 4.6, 1.7)
    on shortened or extra-ball overs.
    """
    for pp in innings.get("powerplays") or []:
        if pp.get("type") == "mandatory":
            return int(float(pp.get("to", 5.6)))
    return 5


def _phase_for(over: int, pp_last_over: int, total_overs: int) -> str:
    if over <= pp_last_over:
        return "pp"
    if over >= total_overs - DEATH_OVERS:
        return "death"
    return "mid"


def _classify(delivery: dict, striker: str) -> tuple[str, str | None, bool]:
    """-> (outcome class, wicket kind, drop_row).

    Ordering matters: a wide is not faced by the batter at all, and a no-ball
    diverts the game into its free-hit branch, so both take precedence over
    whatever happened off the bat.
    """
    extras = delivery.get("extras") or {}
    wickets = delivery.get("wickets") or []

    for w in wickets:
        if w.get("kind") in RETIREMENTS:
            return "0", None, True          # not a contest outcome; drop the row

    if extras.get("wides"):
        return "wide", None, False
    if extras.get("noballs"):
        return "noball", None, False

    for w in wickets:
        kind = w.get("kind")
        out_player = w.get("player_out")
        if kind == "run out":
            # Credited to whoever was run out, not to the bowler-vs-striker
            # contest. A run out of the striker still ends his innings, so it is
            # labelled Out (with the runs already taken discarded -- see the
            # ~0.3 runs/innings undercount noted in the plan); a run out of the
            # NON-striker is not the striker's outcome at all.
            if out_player == striker:
                return "Out", "run out", False
            continue
        if kind in STRIKER_DISMISSALS and out_player == striker:
            return "Out", kind, False

    runs = delivery["runs"]["batter"]
    if runs == 5:
        return "4", None, False             # not modelled; folded into the nearest class
    if runs > 6:
        return "6", None, False
    return str(runs), None, False


def _run_out_victim(delivery: dict) -> str | None:
    for w in delivery.get("wickets") or []:
        if w.get("kind") == "run out":
            return w.get("player_out")
    return None


def _any_dismissal(delivery: dict) -> tuple[str | None, str | None]:
    for w in delivery.get("wickets") or []:
        if w.get("kind") in RETIREMENTS:
            return w.get("player_out"), w.get("kind")
        return w.get("player_out"), w.get("kind")
    return None, None


# --- the replay ------------------------------------------------------------

def replay_innings(innings: dict, meta: dict, innings_no: int) -> InningsRecord | None:
    """Walk one innings' deliveries in order, deriving full state for each ball."""
    if innings.get("super_over"):
        return None
    if innings.get("miscounted_overs"):
        return None

    registry: dict = meta["registry"]
    total_overs = meta["total_overs"]
    pp_last = _phase_bounds(innings, total_overs)
    target = (innings.get("target") or {}).get("runs") if innings_no == 2 else None

    rec = InningsRecord(
        match_id=meta["match_id"],
        season=meta["season"],
        venue=meta["venue"],
        innings_no=innings_no,
        batting_team=innings.get("team", ""),
        bowling_team=meta["other_team"](innings.get("team", "")),
        target=target,
        lineup=[],
        bowler_by_over=[],
    )

    score = 0
    wickets = 0
    legal = 0
    total_balls = total_overs * 6

    bat_runs: dict[str, int] = {}
    bat_balls: dict[str, int] = {}
    bat_position: dict[str, int] = {}
    bowl_balls: dict[str, int] = {}
    bowl_runs: dict[str, int] = {}
    bowl_wkts: dict[str, int] = {}
    spell_last_over: dict[str, int] = {}
    spell_len: dict[str, int] = {}

    partnership_runs = 0
    partnership_balls = 0
    free_hit = False

    def see(name: str) -> None:
        if name not in bat_position:
            bat_position[name] = len(bat_position) + 1
            bat_runs[name] = 0
            bat_balls[name] = 0
            rec.lineup.append(name)

    for over_block in innings.get("overs") or []:
        over = over_block["over"]
        deliveries = over_block.get("deliveries") or []
        if not deliveries:
            continue

        bowler = deliveries[0].get("bowler", "")
        while len(rec.bowler_by_over) <= over:
            rec.bowler_by_over.append(bowler)
        rec.bowler_by_over[over] = bowler

        # spell tracking: an unbroken run of overs by the same bowler
        if spell_last_over.get(bowler) == over - 2:
            # bowlers alternate ends, so consecutive overs for one bowler are 2 apart
            spell_len[bowler] = spell_len.get(bowler, 0) + 1
        else:
            spell_len[bowler] = 1
        spell_last_over[bowler] = over

        ball_in_over = 0
        for delivery in deliveries:
            striker = delivery.get("batter", "")
            non_striker = delivery.get("non_striker", "")
            see(striker)
            see(non_striker)

            outcome, kind, drop = _classify(delivery, striker)
            extras = delivery.get("extras") or {}
            is_legal = not extras.get("wides") and not extras.get("noballs")
            runs_batter = delivery["runs"]["batter"]
            runs_total = delivery["runs"]["total"]

            if is_legal:
                ball_in_over += 1

            if not drop:
                rec.balls.append(BallRecord(
                    match_id=rec.match_id,
                    season=rec.season,
                    venue=rec.venue,
                    innings_no=innings_no,
                    batting_team=rec.batting_team,
                    bowling_team=rec.bowling_team,
                    batter=striker,
                    batter_id=registry.get(striker, striker),
                    bowler=bowler,
                    bowler_id=registry.get(bowler, bowler),
                    non_striker=non_striker,
                    non_striker_id=registry.get(non_striker, non_striker),
                    over=over,
                    ball_in_over=max(1, ball_in_over),
                    legal_index=legal,
                    phase=_phase_for(over, pp_last, total_overs),
                    score=score,
                    wickets=wickets,
                    balls_remaining=max(0, total_balls - legal),
                    target=target,
                    striker_runs=bat_runs.get(striker, 0),
                    striker_balls=bat_balls.get(striker, 0),
                    striker_position=bat_position.get(striker, 11),
                    partnership_runs=partnership_runs,
                    partnership_balls=partnership_balls,
                    bowler_balls=bowl_balls.get(bowler, 0),
                    bowler_runs=bowl_runs.get(bowler, 0),
                    bowler_wickets=bowl_wkts.get(bowler, 0),
                    over_in_spell=spell_len.get(bowler, 1),
                    free_hit=free_hit,
                    outcome=outcome,
                    runs_batter=runs_batter,
                    runs_total=runs_total,
                    is_legal=is_legal,
                    wicket_kind=kind,
                ))

            # --- advance state (order matters: records above are pre-ball) ---
            score += runs_total
            partnership_runs += runs_total
            bowl_runs[bowler] = bowl_runs.get(bowler, 0) + runs_total - (
                extras.get("byes", 0) + extras.get("legbyes", 0)
            )

            if is_legal:
                legal += 1
                partnership_balls += 1
                bat_balls[striker] = bat_balls.get(striker, 0) + 1
                bowl_balls[bowler] = bowl_balls.get(bowler, 0) + 1
            elif extras.get("noballs"):
                # a no-ball is faced by the batter even though it isn't a legal
                # delivery for the over count
                bat_balls[striker] = bat_balls.get(striker, 0) + 1

            if not extras.get("wides"):
                bat_runs[striker] = bat_runs.get(striker, 0) + runs_batter

            free_hit = bool(extras.get("noballs"))

            victim, vkind = _any_dismissal(delivery)
            if victim and vkind not in RETIREMENTS:
                wickets += 1
                partnership_runs = 0
                partnership_balls = 0
                if vkind != "run out":
                    bowl_wkts[bowler] = bowl_wkts.get(bowler, 0) + 1

    # pad out the named XI: everyone who didn't get to bat still exists, and a
    # replay of this innings needs a full order to know what "all out" means
    for name in meta["squads"].get(rec.batting_team, ()):
        if name not in bat_position:
            rec.lineup.append(name)

    rec.total = score
    rec.wickets = wickets
    rec.legal_balls = legal
    return rec


def iter_innings(match_glob: str = DEFAULT_MATCH_GLOB, limit: int | None = None):
    """Yield `InningsRecord` for every eligible innings, oldest match first."""
    paths = sorted(glob.glob(match_glob))
    if limit:
        paths = paths[:limit]

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue

        info = data.get("info") or {}
        if not match_is_eligible(info):
            continue

        teams = info.get("teams") or ["", ""]
        # Season comes from the match DATE, not the `season` label.
        #
        # The label is ambiguous: it is a plain year most of the time ("2017") but
        # a split year for tournaments that straddle a new year ("2007/08"). Taking
        # the part before the slash collides -- "2009" (IPL 2009) and "2009/10"
        # (IPL 2010) both reduce to 2009, silently merging two whole tournaments
        # into one season with double the balls. "2007/08" likewise reports 2007
        # for a tournament played entirely in 2008.
        #
        # The date is unambiguous and present on every match, so use it.
        dates = info.get("dates") or []
        season = int(str(dates[0])[:4]) if dates else 0

        meta = {
            "match_id": os.path.splitext(os.path.basename(path))[0],
            "season": season,
            "venue": info.get("venue", ""),
            "registry": (info.get("registry") or {}).get("people") or {},
            "squads": info.get("players") or {},
            "total_overs": info.get("overs", 20),
            "other_team": lambda t, _teams=teams: (
                _teams[1] if t == _teams[0] else _teams[0]
            ),
        }

        for i, innings in enumerate(data.get("innings") or [], start=1):
            if i > 2:
                break        # super overs and beyond
            rec = replay_innings(innings, meta, i)
            if rec is not None and rec.balls:
                yield rec


def iter_balls(match_glob: str = DEFAULT_MATCH_GLOB, limit: int | None = None):
    for innings in iter_innings(match_glob, limit):
        yield from innings.balls


if __name__ == "__main__":
    import collections
    import time

    t0 = time.time()
    n_innings = 0
    n_balls = 0
    classes = collections.Counter()
    kinds = collections.Counter()
    totals = []

    for innings in iter_innings():
        n_innings += 1
        n_balls += len(innings.balls)
        totals.append(innings.total)
        for b in innings.balls:
            classes[b.outcome] += 1
            if b.wicket_kind:
                kinds[b.wicket_kind] += 1

    print(f"{n_innings} innings, {n_balls} balls in {time.time() - t0:.1f}s")
    print("\noutcome distribution:")
    for c in CLASSES:
        print(f"  {c:<8} {classes[c]:>7}  {100 * classes[c] / n_balls:5.2f}%")
    print("\nwicket kinds:")
    for k, v in kinds.most_common():
        print(f"  {k:<22} {v:>6}")
    mean = sum(totals) / len(totals)
    var = sum((t - mean) ** 2 for t in totals) / len(totals)
    print(f"\ninnings total: mean {mean:.1f}  sd {var ** 0.5:.1f}")
