"""Run N innings through any ball function and collect innings-level outcomes.

`ball_fn` has exactly the signature of `src.engine.simulator.calculate_single_ball`,
so the same harness scores the current engine today and the learned model later
with no changes.

Innings are replayed from REAL matches -- real XIs, real batting orders, real
bowler-per-over allocations. Simulating against a synthetic squad sampler would
measure the sampler rather than the engine.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from src.engine.simulator import EXTRAS_PROB, WIDE_SHARE_OF_EXTRAS
from ml.harness.stats import OVERS, InningsOutcome
from ml.runtime import features as F
from ml.runtime import players as P

MAX_WICKETS = 10


@dataclass(frozen=True)
class RoleMix:
    """How often each role gets picked, per phase.

    DESIGNED, NOT MEASURED. Cricsheet records what happened, never what anyone was
    trying to do, so there is no ground truth for these. They exist so the baseline
    is scored under a plausible mix of player choices rather than under an
    all-neutral setting that no real game would produce.
    """

    # phase -> (attack, rotate, defend)
    bat: dict = None
    # phase -> (attack, contain, defend)
    bowl: dict = None

    @staticmethod
    def realistic() -> "RoleMix":
        return RoleMix(
            bat={
                "pp":    (0.30, 0.50, 0.20),
                "mid":   (0.25, 0.55, 0.20),
                "death": (0.60, 0.30, 0.10),
            },
            bowl={
                "pp":    (0.40, 0.50, 0.10),
                "mid":   (0.30, 0.60, 0.10),
                "death": (0.30, 0.40, 0.30),
            },
        )

    @staticmethod
    def neutral() -> "RoleMix":
        """Everyone rotates / contains. Isolates the base distribution."""
        return RoleMix(
            bat={p: (0.0, 1.0, 0.0) for p in ("pp", "mid", "death")},
            bowl={p: (0.0, 1.0, 0.0) for p in ("pp", "mid", "death")},
        )


_BAT_ROLES = ("attack", "rotate", "defend")
_BOWL_ROLES = ("attack", "contain", "defend")
# role -> pseudo-intent, so the Trap gambit's aggression read still works
_ROLE_INTENT = {"attack": 70, "rotate": 50, "defend": 35}


def _pick(rng: random.Random, options, weights):
    return rng.choices(options, weights=weights, k=1)[0]


def venue_rates(plans) -> dict:
    """venue -> (runs/ball, wickets/ball), with a None key for the league fallback.

    Mirrors the ETL's venue feature so the model sees the same quantity in
    simulation that it was trained on. Not leave-one-match-out here -- at play time
    there is no "current match" to leave out, which is exactly the situation.
    """
    agg: dict = {}
    tot = [0, 0, 0]
    for p in plans:
        a = agg.setdefault(p.venue, [0, 0, 0])
        for acc in (a, tot):
            acc[0] += p.legal_balls
            acc[1] += p.total
            acc[2] += p.wickets
    league = (tot[1] / max(1, tot[0]), tot[2] / max(1, tot[0]))
    out = {None: league}
    for v, (b, r, w) in agg.items():
        out[v] = (r / b, w / b) if b >= 600 else league
    return out


def simulate_innings(
    lineup: list[dict],
    bowler_by_over: list[dict],
    ball_fn,
    league_avg: dict,
    *,
    target: int | None = None,
    rng: random.Random,
    role_mix: RoleMix,
    pitch: str | None = None,
    extras_fn=None,
    use_roles: bool = True,
    venue_rpb: float = 1.358,
    venue_wpb: float = 0.0493,
    day_sigma: float = 0.0,
) -> InningsOutcome:
    """One innings. `lineup` and `bowler_by_over` are raw player records.

    `extras_fn(striker, bowler, ctx) -> (p_wide, p_noball)` overrides the engine's
    flat 4% / 70-30 split. Extras sit outside the weight pipeline in the classic
    engine, so they can't ride in through `ball_fn` and need their own hook.

    `ctx` here is the SAME situation context handed to `ball_fn` for the
    legal-delivery case (over, wickets, score, the specific bowler's stats...),
    built once per ball attempt before it's known whether the ball is an extra.
    A model-backed `extras_fn` needs that full picture, not just the over number
    -- wides are heavily bowler-dependent, and a flat per-over average can't tell
    a wild bowler from a metronomic one.
    """
    bat_objs = [P.make_batter(r) for r in lineup]
    out = InningsOutcome(lineup_size=len(bat_objs))
    scores = [0] * len(bat_objs)
    balls_faced = [0] * len(bat_objs)

    striker, non_striker, next_in = 0, 1, 2
    runs = wickets = legal = 0
    free_hit = False
    partnership_balls = 0
    bowler_balls: dict[str, int] = {}
    spell_last_over: dict[str, int] = {}
    spell_len: dict[str, int] = {}
    # a per-innings day factor: drawn once and HELD. Persistent randomness is the
    # only kind that moves innings-total variance -- an iid per-ball jitter of the
    # same size washes out as sqrt(120) instead of scaling with 120.
    day_factor = rng.gauss(0.0, day_sigma) if day_sigma else 0.0

    for over in range(OVERS):
        if wickets >= min(MAX_WICKETS, len(bat_objs) - 1):
            break
        if target is not None and runs >= target:
            break

        b_rec = bowler_by_over[over % len(bowler_by_over)]
        phase = P.phase_key(over)
        bowl_role = _pick(rng, _BOWL_ROLES, role_mix.bowl[phase])
        bowler = P.make_bowler(b_rec, intent=50)
        b_grid = P.bowl_grid(b_rec, over, bowl_role)

        if spell_last_over.get(bowler.name) == over - 2:
            spell_len[bowler.name] = spell_len.get(bowler.name, 0) + 1
        else:
            spell_len[bowler.name] = 1
        spell_last_over[bowler.name] = over

        # roles follow the person, chosen once per over -- as the server does
        roles = {i: _pick(rng, _BAT_ROLES, role_mix.bat[phase]) for i in (striker, non_striker)}
        wkts_this_over = 0
        balls_this_over = 0

        while balls_this_over < 6:
            if wickets >= min(MAX_WICKETS, len(bat_objs) - 1):
                break
            if target is not None and runs >= target:
                break

            s_obj = bat_objs[striker]

            # Situation context: everything the model needs, built once per ball
            # ATTEMPT, before we know whether it turns out to be an extra. Roles
            # and gambits don't belong here -- they're a layer applied only to the
            # 7 legal-delivery buckets, never to the extras decision, so the model
            # genuinely doesn't need them to predict wide/no-ball.
            sit_ctx = {
                "over_num": over,
                "ball_in_over": balls_this_over + 1,
                "score": runs,
                "wickets": wickets,
                "balls_remaining": max(0, OVERS * 6 - legal),
                "innings_no": 2 if target else 1,
                "target": target,
                "striker_balls": balls_faced[striker],
                "striker_position": striker + 1,
                "partnership_balls": partnership_balls,
                "bowler_balls": bowler_balls.get(bowler.name, 0),
                "over_in_spell": spell_len.get(bowler.name, 1),
                "bat_career_balls": s_obj.career_balls,
                "bowl_career_balls": bowler.legal_balls,
                # from the RECORD via model_ovr, not from the Batter object: on an
                # era pool `Batter.ovr` carries the derived rating, which the model
                # must never see (it is derived FROM the model). See
                # features.model_ovr.
                "ns_ovr": (F.model_ovr(lineup[non_striker])
                           if non_striker < len(lineup) else 55.0),
                "ns_sr": bat_objs[non_striker].sr if non_striker < len(bat_objs) else 120.0,
                "venue_rpb": venue_rpb,
                "venue_wpb": venue_wpb,
                "day_factor": day_factor,
            }

            # --- extras: rolled outside the weight pipeline, exactly as the engine
            # does, but now with the model's own context-aware prediction (bowler
            # identity included) when a model-backed extras_fn is supplied
            if extras_fn is not None:
                p_wide, p_nb = extras_fn(s_obj, bowler, sit_ctx)
                roll = rng.random()
                hit, is_wide = roll < p_wide + p_nb, roll < p_wide
            else:
                # matches whatever the classic engine itself currently does --
                # imported, not duplicated, so this can't drift out of sync with it
                hit = rng.random() < EXTRAS_PROB
                is_wide = rng.random() < WIDE_SHARE_OF_EXTRAS if hit else False

            if not free_hit and hit:
                runs += 1
                out.runs_by_over[over] += 1
                out.counts["wide" if is_wide else "noball"] = (
                    out.counts.get("wide" if is_wide else "noball", 0) + 1
                )
                if not is_wide:
                    free_hit = True
                continue

            s_rec = lineup[striker]
            bat_role = roles.get(striker) or _pick(rng, _BAT_ROLES, role_mix.bat[phase])
            roles[striker] = bat_role

            ctx = dict(sit_ctx)
            ctx.update({
                "pitch": pitch,
                "bowler_style": bowler.style,
                "attack_gambit": False,
                "trap_gambit": False,
                "striker_intent": _ROLE_INTENT[bat_role],
                # bat_role=None makes both the classic engine and the adapter skip
                # Stage 4/5 entirely -- the diagnostic path
                "bat_role": bat_role if use_roles else None,
                "bat_grid": P.bat_grid(s_rec, over, bat_role),
                "bowl_role": bowl_role if use_roles else None,
                "bowl_grid": b_grid,
                "wickets_this_over": wkts_this_over,
            })
            outcome = ball_fn(s_obj, bowler, league_avg, ctx)

            balls_this_over += 1
            legal += 1
            balls_faced[striker] += 1
            partnership_balls += 1
            bowler_balls[bowler.name] = bowler_balls.get(bowler.name, 0) + 1
            out.balls_by_over[over] += 1

            if outcome == "Out" and free_hit:
                outcome = "0"          # no dismissal on a free hit
            free_hit = False

            if outcome == "Out":
                wickets += 1
                wkts_this_over += 1
                out.counts["Out"] = out.counts.get("Out", 0) + 1
                out.wickets_by_over[over] += 1
                if next_in < len(bat_objs):
                    striker = next_in
                    next_in += 1
                    partnership_balls = 0
                    roles[striker] = _pick(rng, _BAT_ROLES, role_mix.bat[phase])
                else:
                    break
                continue

            r = int(outcome)
            runs += r
            scores[striker] += r
            out.runs_by_over[over] += r
            # '5's are folded into the '4' bucket to match the replay's classes;
            # the runs themselves are counted in full
            out.counts["4" if r == 5 else outcome] = (
                out.counts.get("4" if r == 5 else outcome, 0) + 1
            )
            if r % 2 == 1:
                striker, non_striker = non_striker, striker

        striker, non_striker = non_striker, striker      # end of over

    out.total = runs
    out.wickets = wickets
    out.legal_balls = legal
    out.batter_scores = [s for i, s in enumerate(scores) if i < next_in]
    out.chased = target is not None and runs >= target
    return out


def run_batch(
    innings_plans: list,
    ball_fn,
    *,
    n: int = 10_000,
    seed: int = 0,
    role_mix: RoleMix | None = None,
    pitch: str | None = None,
    extras_fn=None,
    use_roles: bool = True,
    day_sigma: float = 0.0,
    era_id: str | None = None,
    progress_every: int = 2000,
) -> list[InningsOutcome]:
    """Cycle through real innings plans until `n` simulated innings are done.

    `era_id` selects which player pool the lineups are built from -- an era's
    records carry that era's own stats, grids and (once derived) OVRs.
    """
    league_avg = P.league_avg()
    by_name = P.load_players(era_id)
    role_mix = role_mix or RoleMix.realistic()
    rng = random.Random(seed)
    # calculate_single_ball samples from the GLOBAL random module, so seed it too
    random.seed(seed)

    venue = venue_rates(innings_plans)

    usable = []
    for plan in innings_plans:
        lineup = [by_name[nm] for nm in plan.lineup if nm in by_name][:11]
        overs = [by_name[nm] for nm in plan.bowler_by_over if nm in by_name]
        # a full XI is required: the all-out threshold is len(lineup) - 1, so a
        # short lineup silently redefines what bowling a side out means
        if len(lineup) == 11 and len(overs) >= 4:
            usable.append((lineup, overs, plan.target,
                           *venue.get(plan.venue, venue[None])))
    if not usable:
        raise RuntimeError("no usable innings plans -- name join with players_historical failed")

    results = []
    for i in range(n):
        lineup, overs, target, v_rpb, v_wpb = usable[i % len(usable)]
        results.append(simulate_innings(
            lineup, overs, ball_fn, league_avg,
            target=target, rng=rng, role_mix=role_mix, pitch=pitch,
            extras_fn=extras_fn, use_roles=use_roles,
            venue_rpb=v_rpb, venue_wpb=v_wpb, day_sigma=day_sigma,
        ))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    ... {i + 1}/{n}", flush=True)
    return results
