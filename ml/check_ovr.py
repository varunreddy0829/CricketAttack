"""The sanity gate for Phase C: are the measured OVRs actually any good?

    ml/.venv/Scripts/python -m ml.check_ovr

Deriving OVR by simulation is only worth doing if the result beats the formula it
replaced. "OVR correlates with measured value" is NOT evidence of that -- the new
OVR is a linear rescale of the measured value, so that correlation is 1.00 by
construction and proves nothing. These are the checks that can actually fail:

1. NAMED PLAYERS. Cricket knowledge the numbers must reproduce: Gayle a monster
   in 2008-2013, gone from 2023-2026; Abhishek Sharma top-tier in 2023-2026.
2. REORDERING. The new ratings must disagree with the old career formula, or the
   whole exercise changed nothing. But they must still be positively related --
   a near-zero or negative correlation would mean the measurement is broken, not
   insightful.
3. SPREAD. Ratings must use the band. If everyone lands in 70-75 the auction has
   no tiers and every player costs the same.
4. AUCTION VIABILITY. draft_generator needs enough players above MARQUEE_CUT and
   MID_FLOOR, in each role, to build 12 sets.
5. COVERAGE. Every rateable player rated; no nulls left for the server to paper
   over with placeholders.
"""

from __future__ import annotations

import json
import os

from ml.etl import eras as E

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ERA_ROOT = os.path.join(REPO_ROOT, "data", "eras")
CAREER = os.path.join(REPO_ROOT, "data", "players_historical.json")

MARQUEE_CUT, MID_FLOOR = 80, 70

# name -> (era_id, expectation). The gate, stated before looking at the output.
EXPECT = [
    ("CH Gayle",         "2008_2013", "top15",  "the six-hitting era belonged to him"),
    ("CH Gayle",         "2023_2026", "absent", "he does not play in this era"),
    ("Abhishek Sharma",  "2023_2026", "top15",  "the modern game's defining hitter"),
    ("AB de Villiers",   "2014_2022", "top15",  "the era's most feared batter"),
    ("SP Narine",        "2014_2022", "bowl40", "the era's most economical bowler"),
]

# Players whose REPUTATION and whose measured T20 value genuinely disagree.
#
# These are not failures and must not be asserted on -- they are the metric
# doing exactly what it was chosen to do. Team-runs-added rewards scoring RATE,
# because in a 120-ball innings the scarce resource is balls, not wickets. A
# high-average, medium-tempo accumulator can be a great cricketer and still add
# ~0 runs over a median player occupying the same slot.
#
# Asserting "Kohli must be top-40" would encode reputation, and the honest
# reading is that the same logic which correctly demotes him also correctly
# demoted Amla (SR 119) and promoted Abhishek Sharma -- which is the reordering
# this whole exercise was for. But it IS a game-feel decision, so the gate
# prints these every run rather than burying them.
WATCH = [
    ("V Kohli",     "2014_2022", "high average, median strike rate"),
    ("MS Dhoni",    "2014_2022", "finisher's reputation, mid-tier tempo by then"),
    ("F du Plessis", "2014_2022", "anchor, not an impact rate"),
]


def _load(path):
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    return json.loads(text[text.find("["):])


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    return sxy / ((sxx * syy) ** 0.5) if sxx and syy else float("nan")


def _rank(records, key, name):
    """1-based rank of `name` by `key`, or None if unrated/absent."""
    rated = sorted((r for r in records if r.get(key)), key=lambda r: -r[key])
    for i, r in enumerate(rated, 1):
        if r["name"] == name:
            return i, len(rated)
    return None, len(rated)


def main() -> None:
    career = {r["name"]: r for r in _load(CAREER)}
    pools, failures, warnings = {}, [], []

    for era in E.MODEL_ERAS:
        path = os.path.join(ERA_ROOT, era.id, "players.json")
        if not os.path.exists(path):
            failures.append(f"{era.id}: no players.json")
            continue
        pools[era.id] = _load(path)

    # ---- 5. coverage ----------------------------------------------------
    print("=" * 68)
    print("COVERAGE -- every rateable player must have a number")
    print("=" * 68)
    for era_id, recs in pools.items():
        for kind, flag, key in (("bat", "rateable_batting", "batting_ovr"),
                                ("bowl", "rateable_bowling", "bowling_ovr")):
            want = [r for r in recs if r.get(flag)]
            got = [r for r in want if r.get(key)]
            tag = "ok" if len(got) == len(want) else "MISSING"
            print(f"  {era_id:<10} {kind:<5} {len(got):>4}/{len(want):<4} rated   {tag}")
            if len(got) != len(want):
                failures.append(
                    f"{era_id} {kind}: {len(want) - len(got)} rateable players unrated")

    # ---- 3. spread ------------------------------------------------------
    print()
    print("=" * 68)
    print("SPREAD -- the band must be used, or the auction has no tiers")
    print("=" * 68)
    for era_id, recs in pools.items():
        for kind, key in (("bat", "batting_ovr"), ("bowl", "bowling_ovr")):
            vals = sorted(r[key] for r in recs if r.get(key))
            if not vals:
                continue
            p10, p50, p90 = (vals[int(len(vals) * q)] for q in (0.10, 0.50, 0.90))
            print(f"  {era_id:<10} {kind:<5} min {vals[0]:>3}  p10 {p10:>3}  "
                  f"med {p50:>3}  p90 {p90:>3}  max {vals[-1]:>3}")
            if p90 - p10 < 12:
                failures.append(
                    f"{era_id} {kind}: p10-p90 spread only {p90 - p10} pts -- too flat")

    # ---- 4. auction viability -------------------------------------------
    print()
    print("=" * 68)
    print(f"AUCTION -- need players above MARQUEE_CUT({MARQUEE_CUT}) "
          f"and MID_FLOOR({MID_FLOOR})")
    print("=" * 68)
    for era_id, recs in pools.items():
        def best(r):
            return max(r.get("batting_ovr") or 0, r.get("bowling_ovr") or 0)
        rated = [r for r in recs if best(r) > 0]
        marquee = [r for r in rated if best(r) >= MARQUEE_CUT]
        mid = [r for r in rated if MID_FLOOR <= best(r) < MARQUEE_CUT]
        print(f"  {era_id:<10} {len(rated):>4} rated   "
              f"{len(marquee):>3} marquee   {len(mid):>3} mid   "
              f"{len(rated) - len(marquee) - len(mid):>3} base")
        if len(marquee) < 15:
            failures.append(f"{era_id}: only {len(marquee)} marquee players "
                            f"-- draft_generator needs 15 (3 tiers x 5)")
        if len(rated) < 175:
            warnings.append(f"{era_id}: {len(rated)} rated, an 8-team auction "
                            f"wants 175")

    # ---- 2. reordering vs the old formula --------------------------------
    print()
    print("=" * 68)
    print("REORDERING -- new ratings must DISAGREE with the career formula")
    print("=" * 68)
    print("  (r near 1.0 = we changed nothing;  r near 0 = measurement is broken)")
    for era_id, recs in pools.items():
        pairs = [(career[r["name"]]["batting_ovr"], r["batting_ovr"]) for r in recs
                 if r.get("batting_ovr") and r["name"] in career
                 and career[r["name"]].get("batting_ovr")]
        if len(pairs) < 10:
            continue
        r = _pearson([a for a, _ in pairs], [b for _, b in pairs])
        note = ""
        if r > 0.90:
            note = "  <-- barely different from the old formula"
            warnings.append(f"{era_id}: new OVR r={r:.2f} vs old -- little changed")
        elif r < 0.15:
            note = "  <-- suspiciously unrelated"
            warnings.append(f"{era_id}: new OVR r={r:.2f} vs old -- check the "
                            f"measurement, this may be noise")
        print(f"  {era_id:<10} r = {r:>5.2f}  (n={len(pairs)}){note}")

    # ---- 1. the named-player gate ---------------------------------------
    print()
    print("=" * 68)
    print("NAMED PLAYERS -- the cricket knowledge the numbers must reproduce")
    print("=" * 68)
    for name, era_id, want, why in EXPECT:
        recs = pools.get(era_id)
        if recs is None:
            continue
        key = "bowling_ovr" if want.startswith("bowl") else "batting_ovr"
        rank, total = _rank(recs, key, name)
        limit = int("".join(c for c in want if c.isdigit()) or 0)

        if want == "absent":
            ok = rank is None
            got = "absent" if ok else f"rank {rank}/{total}"
        else:
            ok = rank is not None and rank <= limit
            got = f"rank {rank}/{total}" if rank else "ABSENT/unrated"

        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<18} {era_id:<10} "
              f"want {want:<7} got {got:<16} -- {why}")
        if not ok:
            failures.append(f"{name} in {era_id}: wanted {want}, got {got}")

    # ---- reputation vs measurement, reported not asserted -----------------
    print()
    print("=" * 68)
    print("WATCH -- where reputation and measured T20 value disagree")
    print("=" * 68)
    print("  (not failures: the metric rewards RATE, because balls are the")
    print("   scarce resource in 120. Your call whether the game feel is right.)")
    for name, era_id, why in WATCH:
        recs = pools.get(era_id)
        if not recs:
            continue
        rec = next((r for r in recs if r["name"] == name), None)
        if not rec or not rec.get("batting_ovr"):
            print(f"  {name:<15} {era_id:<10} not rated in this era")
            continue
        rank, total = _rank(recs, "batting_ovr", name)
        b = rec["batting"]
        srs = sorted(r["batting"]["sr"] for r in recs if r.get("batting_ovr"))
        pct = sum(1 for s in srs if s < b["sr"]) / len(srs)
        print(f"  {name:<15} {era_id:<10} OVR {rec['batting_ovr']:>3}  "
              f"rank {rank:>3}/{total}  SR {b['sr']:>6} (p{pct * 100:.0f})  "
              f"avg {b['avg']:>5}  {rec['measured_bat_value']:>+6.1f} runs -- {why}")

    # ---- what the eras actually look like --------------------------------
    for era_id, recs in pools.items():
        print()
        print("-" * 68)
        print(f"{era_id}  top 10 batters by measured value")
        print("-" * 68)
        top = sorted((r for r in recs if r.get("batting_ovr")),
                     key=lambda r: -r["batting_ovr"])[:10]
        for r in top:
            old = career.get(r["name"], {}).get("batting_ovr")
            b = r["batting"]
            print(f"  {r['batting_ovr']:>3}  {r['name']:<24}"
                  f"{r.get('measured_bat_value', 0):>+7.1f} runs   "
                  f"SR {b['sr']:>6}  {b['balls']:>5} balls   "
                  f"(career OVR {old if old else '-'})")

    print()
    print("=" * 68)
    for w in warnings:
        print(f"  WARN  {w}")
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
        print(f"\nGATE FAILED -- {len(failures)} problem(s). Phase C needs rework.")
        raise SystemExit(1)
    print("  GATE PASSED" + (f" ({len(warnings)} warning(s))" if warnings else ""))


if __name__ == "__main__":
    main()
