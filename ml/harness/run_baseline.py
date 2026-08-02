"""Phase 0: score the CURRENT, UNMODIFIED engine against real IPL.

    python -m ml.harness.run_baseline [--n 10000] [--neutral]

This is the scoreboard. It touches nothing in src/ -- it imports
`calculate_single_ball` and runs it. Every later phase is measured against the
numbers this prints.

It also settles a standing contradiction in the repo: src/engine/simulator.py's
comment records 151 runs / 37.5% all-out, while CLAUDE.md claims ~170 / ~14%.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from src.engine.simulator import calculate_single_ball
from ml.etl.replay import iter_innings
from ml.harness import compare
from ml.harness.simulate import RoleMix, run_batch
from ml.harness.stats import from_replay, summarize

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")


def real_reference(limit: int | None = None, season_min: int | None = None):
    """Replay every eligible innings and summarize. Returns (summary, plans).

    `season_min` restricts BOTH the targets and the replayed lineups to recent
    cricket. This matters more than it looks: 2008-2015 ran 7.50 rpo and 2023-2026
    runs 9.01 (ml/etl/season_shift.py), and 33% of modern innings pass 200 against
    13% all-time. Calibrating against the all-time average aims the simulator at a
    game that stopped being played years ago.

    Targets and plans are filtered together on purpose -- scoring modern targets
    while replaying 2010 XIs and 2010 bowling plans would be measuring a mismatch.
    """
    plans = list(iter_innings(limit=limit))
    if season_min is not None:
        plans = [p for p in plans if p.season >= season_min]
    outcomes = [from_replay(p) for p in plans]
    return summarize(outcomes), plans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000, help="innings to simulate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--neutral", action="store_true",
                    help="all batters rotate / all bowlers contain (isolates the base)")
    ap.add_argument("--limit", type=int, default=None, help="cap matches read (for a quick run)")
    args = ap.parse_args()

    os.makedirs(ARTIFACTS, exist_ok=True)

    print("[1/2] replaying real matches ...", flush=True)
    t0 = time.time()
    real, plans = real_reference(limit=args.limit)
    print(f"      {real['n_innings']} real innings in {time.time() - t0:.1f}s")

    with open(os.path.join(ARTIFACTS, "reference_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(real, fh, indent=2)

    mix = RoleMix.neutral() if args.neutral else RoleMix.realistic()
    label = "neutral" if args.neutral else "realistic"
    print(f"\n[2/2] simulating {args.n} innings through the CURRENT engine "
          f"(role mix: {label}) ...", flush=True)
    t0 = time.time()
    sims = run_batch(plans, calculate_single_ball, n=args.n, seed=args.seed, role_mix=mix)
    dt = time.time() - t0
    balls = sum(s.legal_balls for s in sims)
    print(f"      {args.n} innings / {balls} balls in {dt:.1f}s "
          f"({1e6 * dt / max(1, balls):.0f} us/ball)")

    sim = summarize(sims)
    with open(os.path.join(ARTIFACTS, f"baseline_{label}.json"), "w", encoding="utf-8") as fh:
        json.dump(sim, fh, indent=2)

    compare.report(real, sim, title="classic")


if __name__ == "__main__":
    main()
