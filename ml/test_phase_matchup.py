"""Did the interaction features actually work? The falsifiable test.

    ml/.venv/Scripts/python -m ml.test_phase_matchup

The new model added features that only exist because the backbone is ADDITIVE:
a number sitting in a player's anchor is identical in over 1 and over 19, and
identical against spin and pace, so "this player is a death-overs hitter" or
"this player can't play spin" was structurally unlearnable. Those facts now enter
as context features resolved per ball.

That claim is testable, and these are the two tests it has to pass. Both are
stated as real-world targets BEFORE looking at the output, and both can fail:

  PHASE   AB de Villiers really struck at 128.7 in the powerplay and 229.4 at the
          death in 2014-2022 -- a 101-point acceleration. RA Tripathi went 141.7
          to 143.5, a flat 2. If the model shows both as flat, the phase feature
          did nothing.

  MATCHUP Maxwell really struck at 165.7 vs spin and 146.5 vs pace; Jadeja 99.1
          vs spin and 143.2 vs pace. Against SPIN that is a 67-point chasm; against
          PACE the same two players are nearly level (3.9 apart). A model without
          matchup features can only rank them by overall quality and must show
          Maxwell far ahead in BOTH.

This probes the model's probabilities directly rather than simulating innings:
a full simulation confounds the feature under test with lineup, dismissals and
the day factor. Expected runs per ball is the cleanest read of what the model
believes about this exact matchup.

READ THE ABSOLUTE NUMBERS WITH CARE. Every probe here is against a NAMED bowler
-- Bumrah for pace, Rashid Khan for spin -- and both were the best in the era.
So Maxwell reads 134.9 against Rashid where his real figure against ALL spin is
165.7, and that is the model being right rather than wrong: weighted across the
50 spinners he actually faced it comes to 165.2. The gradient by bowler quality
is steep and correct (Maxwell vs Rashid 134.9, vs Tahir 160.5, vs Karanveer
184.4).

Only the DIFFERENCES below are assertions, and they are all within-batter or
between two batters facing the SAME bowler, so bowler quality cancels.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ml.etl import eras as E
from ml.runtime import features as F
from ml.runtime.engine import load_calibration
from ml.runtime.model import OutcomeModel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ERA_ID = "2014_2022"

PHASE_OVERS = {"pp": 2, "mid": 10, "death": 17}
SCORING = ("0", "1", "2", "3", "4", "6")


_JOINT = None


def _joint_counts():
    """batter -> {'spin'|'pace'} -> {phase: balls faced}, from the replay.

    Not stored in the player records because nothing in the MODEL needs the
    crossed distribution -- only this test does, to weight a model probe the
    same way history averaged the real figure.
    """
    global _JOINT
    if _JOINT is not None:
        return _JOINT
    from collections import defaultdict

    from ml.etl.replay import iter_innings
    from src.utils.compile_player_stats import KNOWN_SPINNERS

    era = E.get(ERA_ID)
    j = defaultdict(lambda: {"spin": defaultdict(int), "pace": defaultdict(int),
                             "pos": 0, "wkts": 0, "n": 0})
    for inn in iter_innings():
        if not era.covers(inn.season):
            continue
        for b in inn.balls:
            if b.outcome == "wide":
                continue
            k = "spin" if b.bowler in KNOWN_SPINNERS else "pace"
            e = j[b.batter]
            e[k][F.phase_of(b.over)] += 1
            # where he ACTUALLY batted. Probing a number 7 as though he were a
            # number 3 with two wickets down flatters him by ~25 strike points,
            # uniformly across bowling type -- which is what gave Jadeja a +26
            # error on both spin and pace at once.
            e["pos"] += b.striker_position
            e["wkts"] += b.wickets
            e["n"] += 1
    _JOINT = j
    return j


def _pool():
    with open(os.path.join(REPO_ROOT, "data", "eras", ERA_ID, "players.json"),
              "r", encoding="utf-8") as fh:
        return {p["name"]: p for p in json.load(fh)}


def expected_sr(model, cal, bat_rec, bowl_rec, over: int,
                pos: int = 3, wkts_at: int | None = None) -> float:
    """Model's expected runs/ball x100 for this matchup, averaged over a spread
    of match states so one arbitrary situation can't drive the answer.

    `pos` and `wkts_at` place the batter where he really bats. Left at the
    defaults this over-rates a lower-order player, because the model correctly
    knows a number 3 with two wickets down scores faster than a number 7 with
    six down."""
    is_spin = (bowl_rec or {}).get("bowling_style") == "Spin"
    wkt_states = ((wkts_at - 1, wkts_at, wkts_at + 1) if wkts_at is not None
                  else (1, 3, 5))
    tot, n = 0.0, 0
    for wkts in (max(0, min(9, w)) for w in wkt_states):
        for sb in (2, 12, 30):
            row = F.empty_row()
            F.build_row(
                row,
                over=over, ball_in_over=3, wickets=wkts,
                balls_remaining=120 - over * 6, innings_no=1, score=60 + over * 7,
                target=None, striker_balls=sb, striker_position=pos,
                bowler_balls=6, over_in_spell=2,
                bat_career_balls=(bat_rec.get("batting") or {}).get("balls", 0),
                bowl_career_balls=(bowl_rec.get("bowling") or {}).get("legal_balls", 0),
                ns_sr=140.0, venue_rpb=1.345, venue_wpb=0.0485,
                venue_bdry_share=0.589, venue_type_edge=0.0,
                edges=F.resolve_edges(bat_rec, bowl_rec, over, is_spin),
            )
            p = model.probs(bat_rec["name"], bowl_rec["name"], row)
            runs = sum(float(p[model.ci[c]]) * int(c) for c in SCORING)
            # the calibration multiplier the live engine also applies
            tot += runs * cal["calibration"] * 100
            n += 1
    return tot / n


def main() -> None:
    pool = _pool()
    model = OutcomeModel.load(era_id=ERA_ID)
    cal = load_calibration(era_id=ERA_ID)
    failures = []

    # a representative attack, so the phase test isn't reading one bowler
    pace = pool["JJ Bumrah"]
    spin = pool["Rashid Khan"]

    print("=" * 70)
    print("TEST 1 -- PHASE.  Does the model accelerate a death hitter?")
    print("=" * 70)
    print(f"  vs {pace['name']} throughout, so only the ACCEL column is a claim --")
    print("  the levels are against one elite bowler, not against average pace.")
    print(f"  {'batter':<18}{'pp':>9}{'mid':>9}{'death':>9}{'accel':>10}   real accel")
    phase_res = {}
    for name, real_acc in (("AB de Villiers", 100.7), ("RA Tripathi", 1.9)):
        rec = pool[name]
        vals = {ph: expected_sr(model, cal, rec, pace, ov)
                for ph, ov in PHASE_OVERS.items()}
        acc = vals["death"] - vals["pp"]
        phase_res[name] = acc
        print(f"  {name:<18}{vals['pp']:>9.1f}{vals['mid']:>9.1f}"
              f"{vals['death']:>9.1f}{acc:>+10.1f}   {real_acc:+.1f}")

    abd, trip = phase_res["AB de Villiers"], phase_res["RA Tripathi"]
    if abd < 25:
        failures.append(f"ABD accelerates only {abd:+.1f} -- phase feature is inert")
    if abd - trip < 20:
        failures.append(
            f"ABD and Tripathi accelerate almost identically ({abd:+.1f} vs "
            f"{trip:+.1f}) -- the model is not separating them by phase")

    print()
    print("=" * 70)
    print("TEST 2 -- MATCHUP.  Does spin/pace change WHO is better?")
    print("=" * 70)
    print(f"  vs {spin['name']} (spin) and {pace['name']} (pace) -- both elite, so")
    print("  the levels sit below each batter's figure against an average attack.")
    print(f"  {'batter':<18}{'vs SPIN':>10}{'vs PACE':>10}{'diff':>9}   real diff")
    m_res = {}
    for name, real_diff in (("GJ Maxwell", +19.2), ("RA Jadeja", -44.1)):
        rec = pool[name]
        vs_s = expected_sr(model, cal, rec, spin, 10)
        vs_p = expected_sr(model, cal, rec, pace, 10)
        m_res[name] = (vs_s, vs_p)
        print(f"  {name:<18}{vs_s:>10.1f}{vs_p:>10.1f}{vs_s - vs_p:>+9.1f}"
              f"   {real_diff:+.1f}")

    max_s, max_p = m_res["GJ Maxwell"]
    jad_s, jad_p = m_res["RA Jadeja"]
    gap_spin, gap_pace = max_s - jad_s, max_p - jad_p
    print()
    print(f"  Maxwell - Jadeja  vs spin {gap_spin:+.1f}   vs pace {gap_pace:+.1f}")
    print(f"  real              vs spin    +66.6   vs pace     +3.3")
    if gap_spin - gap_pace < 15:
        failures.append(
            f"the Maxwell-Jadeja gap barely changes with bowling type "
            f"(spin {gap_spin:+.1f} vs pace {gap_pace:+.1f}) -- matchup features "
            f"are not doing their job")

    print()
    print("=" * 70)
    print("TEST 3 -- LEVEL.  Against the WHOLE attack, not one elite bowler.")
    print("=" * 70)
    print("  Tests 1 and 2 deliberately hold the bowler fixed, which depresses")
    print("  every level. This one weights across the real attack, so the")
    print("  absolute numbers become comparable to the historical figures.")
    print("  Weighted over BOTH the attack and the batter's own phase mix -- a")
    print("  historical figure is an average over when he actually batted, and")
    print("  Jadeja takes 59% of his balls at the death (SR 152) against 39% in")
    print("  the middle (SR 98), so probing one over misreads him by 20+ points.")
    level = {}
    print(f"  {'batter':<18}{'type':>7}{'model':>9}{'real':>9}{'err':>8}")
    for name, kind, real in (("GJ Maxwell", "spin", 165.7), ("GJ Maxwell", "pace", 146.5),
                             ("RA Jadeja", "spin", 99.1), ("RA Jadeja", "pace", 143.2)):
        rec = pool[name]
        attack = [p for p in pool.values()
                  if p.get("rateable_bowling")
                  and (p.get("bowling_style") == "Spin") == (kind == "spin")]
        # The JOINT distribution, measured: how many balls this batter faced
        # from THIS bowling type in EACH phase. Weighting by his overall phase
        # mix instead is wrong in a way that looks reasonable -- Jadeja takes
        # 59% of all his balls at the death, but spinners bowl to him in the
        # middle, so his vs-spin figure is a middle-overs number.
        e = _joint_counts()[name]
        joint = e[kind]
        pos = max(1, round(e["pos"] / max(1, e["n"])))
        wk = max(0, round(e["wkts"] / max(1, e["n"])))
        num = den = 0.0
        for ph, ov in PHASE_OVERS.items():
            pw = joint.get(ph, 0)
            if not pw:
                continue
            vals = [expected_sr(model, cal, rec, b, ov, pos=pos, wkts_at=wk)
                    for b in attack]
            w = [b["bowling"]["legal_balls"] for b in attack]
            num += (sum(v * x for v, x in zip(vals, w)) / sum(w)) * pw
            den += pw
        got = num / max(1e-9, den)
        err = got - real
        level[(name, kind)] = (got, real)
        print(f"  {name:<18}{kind:>7}{got:>9.1f}{real:>9.1f}{err:>+8.1f}")

    # THE ASSERTION IS THE SPIN-MINUS-PACE GAP, not the level.
    #
    # A player's historical figure is an average over the exact states he batted
    # in -- position, wickets down, balls already faced, phase, venue. This probe
    # reproduces the first four only coarsely, and what is left shows up as a
    # LEVEL offset: Jadeja reads about +16 on spin AND pace alike, because a
    # synthetic number 7 still scores faster than the real one did. That is a
    # limit of the probe, not a demonstrated model fault -- aggregate scoring is
    # already validated by ml/harness/run_model (0/16 outside tolerance).
    #
    # The gap between the two types cancels every one of those offsets, and it is
    # precisely what the matchup features exist to produce.
    print()
    print(f"  {'batter':<18}{'spin-pace (model)':>19}{'(real)':>10}{'err':>8}")
    for name in ("GJ Maxwell", "RA Jadeja"):
        ms, rs = level[(name, "spin")]
        mp, rp = level[(name, "pace")]
        got, real = ms - mp, rs - rp
        err = got - real
        flag = "" if abs(err) <= 12 else "  <-- off"
        print(f"  {name:<18}{got:>+19.1f}{real:>+10.1f}{err:>+8.1f}{flag}")
        if abs(err) > 12:
            failures.append(
                f"{name}: model spin-pace gap {got:+.1f} against a real "
                f"{real:+.1f} -- the matchup features are not reproducing it")

    print()
    print("=" * 70)
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
        raise SystemExit(1)
    print("  ALL TESTS PASSED")


if __name__ == "__main__":
    main()
