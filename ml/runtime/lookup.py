"""Phase 1 quick win: an empirically measured (over x outcome) base.

No model. Just the real league-average outcome distribution for each over of the
innings, straight out of the ball table, replacing the engine's single global
`BASELINE_WEIGHTS` and its hand-picked `PHASE_EFFECTS` multipliers.

The classic Stage 1/2/3 player ratios still run on top -- they are *relative*
adjustments, so applying them to a per-over league average is exactly what they
were designed for.

This exists to prove the ETL and the harness agree before anything is bet on a
model, and because it captures a real share of the achievable improvement for
about thirty lines.
"""

from __future__ import annotations

import os

import numpy as np

from ml.etl.schema import CLASSES, N_OVERS
from ml.runtime.adapter import ENGINE_KEYS

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
TABLE_PATH = os.path.join(ARTIFACTS, "ball_table.npz")

# '5' is not modelled (~1.5 per 10,000 balls, nearly all overthrows); it is held at
# the engine's own baseline value so the 8-key contract stays intact.
FIVE_WEIGHT = 1.0

LEGAL_CLASSES = ("0", "1", "2", "3", "4", "6", "Out")


def measure(table_path: str = TABLE_PATH):
    """-> (base_by_over: list[dict], extras_by_over: list[(p_wide, p_noball)])"""
    d = np.load(table_path, allow_pickle=True)
    X, y = d["X"], d["y"]
    over = X[:, :N_OVERS].argmax(axis=1)

    ci = {c: i for i, c in enumerate(CLASSES)}
    base, extras = [], []

    for o in range(N_OVERS):
        m = over == o
        yo = y[m]
        n_all = max(1, yo.size)

        p_wide = float((yo == ci["wide"]).sum()) / n_all
        p_nb = float((yo == ci["noball"]).sum()) / n_all
        extras.append((p_wide, p_nb))

        counts = {c: float((yo == ci[c]).sum()) for c in LEGAL_CLASSES}
        legal = max(1.0, sum(counts.values()))
        # scale the seven measured classes into the 1000 - FIVE_WEIGHT budget
        budget = 1000.0 - FIVE_WEIGHT
        w = {c: counts[c] / legal * budget for c in LEGAL_CLASSES}
        w["5"] = FIVE_WEIGHT
        base.append({k: w[k] for k in ENGINE_KEYS})

    return base, extras


def make_provider(base_by_over: list[dict]):
    """base_provider(striker, bowler, ctx) -> 8-key dict. Ignores the players:
    Stage 1/2/3 supply the per-player differentiation on top."""
    tables = [dict(b) for b in base_by_over]

    def provider(striker, bowler, ctx):
        o = ctx.get("over_num", 0) or 0
        return dict(tables[min(max(o, 0), N_OVERS - 1)])

    return provider


def make_extras_fn(extras_by_over):
    """extras_fn(striker, bowler, ctx) -> (p_wide, p_noball). Replaces the engine's
    flat 4% / 70-30 split with the real per-over average -- reality is ~3.75%
    overall at an 89/11 split, and both rates vary sharply by over.

    This is the pre-model version: it only varies by over, not by the specific
    bowler (that needs the model's own context-aware prediction -- see
    ml.harness.run_model.model_extras_fn). Takes the same (striker, bowler, ctx)
    call as that one so `simulate_innings` can call either uniformly; the extra
    args are simply unused here.
    """
    tab = list(extras_by_over)

    def extras_fn(striker, bowler, ctx):
        over = ctx.get("over_num", 0) or 0
        return tab[min(max(over, 0), N_OVERS - 1)]

    return extras_fn


if __name__ == "__main__":
    base, extras = measure()
    print(f"  {'over':>4} {'dot%':>6} {'1%':>6} {'4%':>6} {'6%':>6} {'Out%':>6} "
          f"{'wide%':>6} {'nb%':>5}   runs/ball")
    for o in range(N_OVERS):
        b = base[o]
        rpb = sum(int(k) * b[k] for k in ("1", "2", "3", "4", "5", "6")) / 1000.0
        print(f"  {o:>4} {b['0'] / 10:>6.1f} {b['1'] / 10:>6.1f} {b['4'] / 10:>6.1f} "
              f"{b['6'] / 10:>6.1f} {b['Out'] / 10:>6.1f} "
              f"{100 * extras[o][0]:>6.2f} {100 * extras[o][1]:>5.2f}   {rpb:.3f}")
