"""How much has IPL actually shifted, season by season?

    ml/.venv/Scripts/python -m ml.etl.season_shift

This decides how the train/val/test split should work. If the game has drifted
materially, a chronological split trains the model on cricket that no longer
exists; if it hasn't, chronological is the stricter and more honest test.
"""

from __future__ import annotations

import numpy as np

from ml.etl.schema import CLASSES
from ml.runtime.lookup import TABLE_PATH

RUNS = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "6": 6, "Out": 0}
LEGAL = ("0", "1", "2", "3", "4", "6", "Out")


def main() -> None:
    d = np.load(TABLE_PATH, allow_pickle=True)
    y, season = d["y"], d["season"]
    ci = {c: i for i, c in enumerate(CLASSES)}
    legal_ids = [ci[c] for c in LEGAL]

    hdr = ("season", "balls", "runs/ball", "dot%", "4%", "6%", "Out%", "wide%")
    print("{:>7} {:>7} {:>10} {:>7} {:>6} {:>6} {:>6} {:>6}".format(*hdr))

    rows = []
    for yr in range(2007, 2027):
        m = season == yr
        if m.sum() < 500:
            continue
        yy = y[m]
        lm = np.isin(yy, legal_ids)
        n = int(lm.sum())
        yl = yy[lm]
        rpb = sum(RUNS[c] * int((yl == ci[c]).sum()) for c in LEGAL) / n
        vals = (yr, int(m.sum()), rpb,
                100 * (yl == ci["0"]).sum() / n,
                100 * (yl == ci["4"]).sum() / n,
                100 * (yl == ci["6"]).sum() / n,
                100 * (yl == ci["Out"]).sum() / n,
                100 * (yy == ci["wide"]).sum() / len(yy))
        rows.append(vals)
        print("{:>7} {:>7} {:>10.3f} {:>7.2f} {:>6.2f} {:>6.2f} {:>6.2f} {:>6.2f}"
              .format(*vals))

    early = [r for r in rows if r[0] <= 2015]
    late = [r for r in rows if r[0] >= 2023]
    print()
    for label, group in (("2008-2015", early), ("2023-2026", late)):
        rpb = sum(r[2] for r in group) / len(group)
        six = sum(r[5] for r in group) / len(group)
        dot = sum(r[3] for r in group) / len(group)
        print(f"  {label}:  {rpb:.3f} runs/ball   {dot:.1f}% dots   {six:.2f}% sixes"
              f"   -> {6 * rpb:.2f} rpo")


if __name__ == "__main__":
    main()
