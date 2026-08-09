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


def _pool():
    with open(os.path.join(REPO_ROOT, "data", "eras", ERA_ID, "players.json"),
              "r", encoding="utf-8") as fh:
        return {p["name"]: p for p in json.load(fh)}


def expected_sr(model, cal, bat_rec, bowl_rec, over: int) -> float:
    """Model's expected runs/ball x100 for this matchup, averaged over a spread
    of match states so one arbitrary situation can't drive the answer."""
    is_spin = (bowl_rec or {}).get("bowling_style") == "Spin"
    tot, n = 0.0, 0
    for wkts in (1, 3, 5):
        for sb in (2, 12, 30):
            row = F.empty_row()
            F.build_row(
                row,
                over=over, ball_in_over=3, wickets=wkts,
                balls_remaining=120 - over * 6, innings_no=1, score=60 + over * 7,
                target=None, striker_balls=sb, striker_position=3,
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
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
        raise SystemExit(1)
    print("  BOTH TESTS PASSED")


if __name__ == "__main__":
    main()
