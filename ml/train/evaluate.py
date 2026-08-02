"""Score the learned model and the CLASSIC ENGINE on identical held-out balls.

    ml/.venv/Scripts/python -m ml.train.evaluate

This is the apples-to-apples number. The engine is run as a *predictor*: the same
Stage 1/2/3 + pitch/phase pipeline it uses in play, but emitting probabilities
instead of sampling one draw. Roles are left off for both sides -- the historical
data has no record of what anyone chose, so neither model gets to see them.

Calibration matters more than sharpness here. You will never predict a single ball;
the simulator needs the probabilities to be right *on average within each bucket*,
because errors compound over 120 draws per innings.
"""

from __future__ import annotations

import numpy as np

from src.engine.conditions import apply_conditions
from src.engine.simulator import EXTRAS_PROB, WIDE_SHARE_OF_EXTRAS
from src.engine.stats_calculator import (
    BASELINE_WEIGHTS,
    apply_stage1_ovr,
    apply_stage2_strike_rate_economy,
    apply_stage3_wicket_factor,
)
from ml.etl.schema import CLASSES, N_CLASSES, N_OVERS
from ml.runtime import players as P
from ml.train.backbone import Data, MODEL_PATH, softmax

CI = {c: i for i, c in enumerate(CLASSES)}
# the engine's 8 keys -> our 9 classes; '5' folds into '4', matching the replay
ENGINE_TO_CLASS = {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "4",
                   "6": "6", "Out": "Out"}

# imported, not duplicated: the engine's extras split is now the real measured
# value (see src/engine/simulator.py), so scoring it here can't drift out of sync
WIDE_SHARE = WIDE_SHARE_OF_EXTRAS


def engine_probs(data: Data, mask: np.ndarray) -> np.ndarray:
    """(n, 9) probability matrix from the classic engine."""
    by_name = P.load_players()
    league = P.league_avg()
    names = data.names

    X = data.X[mask]
    overs = X[:, :N_OVERS].argmax(axis=1)
    bat_i, bowl_i = data.bat[mask], data.bowl[mask]

    # cache per (batter, bowler, over) -- the same matchup recurs constantly
    cache: dict = {}
    out = np.zeros((X.shape[0], N_CLASSES), dtype=np.float64)

    p_wide = EXTRAS_PROB * WIDE_SHARE
    p_nb = EXTRAS_PROB * (1.0 - WIDE_SHARE)
    legal_share = 1.0 - EXTRAS_PROB

    for i in range(X.shape[0]):
        key = (bat_i[i], bowl_i[i], overs[i])
        row = cache.get(key)
        if row is None:
            b_rec = by_name.get(str(names[bat_i[i]]))
            w_rec = by_name.get(str(names[bowl_i[i]]))
            if b_rec is None or w_rec is None:
                row = np.full(N_CLASSES, 1.0 / N_CLASSES)
            else:
                batter = P.make_batter(b_rec)
                bowler = P.make_bowler(w_rec)
                w = apply_stage1_ovr(BASELINE_WEIGHTS, batter, bowler)
                w = apply_stage2_strike_rate_economy(w, batter, bowler, league)
                w = apply_stage3_wicket_factor(w, batter, bowler, league)
                w = apply_conditions(w, {"pitch": None, "bowler_style": bowler.style,
                                         "over_num": int(overs[i])})
                row = np.zeros(N_CLASSES)
                total = sum(w.values())
                for k, v in w.items():
                    row[CI[ENGINE_TO_CLASS[k]]] += legal_share * v / total
                row[CI["wide"]] = p_wide
                row[CI["noball"]] = p_nb
            cache[key] = row
        out[i] = row
    return out


def era_breakdown(data, te, y, p_eng, p_mod, p_prior):
    print("\nby era -- the recent rows are the game people actually watch:")
    print(f"  {'era':<10} {'n':>7} {'prior':>9} {'engine':>9} {'model':>9}")
    for label, lo, hi in (("2008-2015", 2007, 2015), ("2016-2022", 2016, 2022),
                          ("2023-2026", 2023, 2026)):
        m = (data.season[te] >= lo) & (data.season[te] <= hi)
        if m.sum() < 200:
            continue
        print(f"  {label:<10} {m.sum():>7} {nll(p_prior[m], y[m]):>9.4f} "
              f"{nll(p_eng[m], y[m]):>9.4f} {nll(p_mod[m], y[m]):>9.4f}")


def model_probs(data: Data, mask: np.ndarray, path: str = MODEL_PATH) -> np.ndarray:
    d = np.load(path, allow_pickle=True)
    E_bat, E_bowl = d["E_bat"], d["E_bowl"]
    z = (d["alpha"]
         + data.X[mask] @ d["B"]
         + E_bat[data.bat[mask]] @ d["V_bat"]
         + E_bowl[data.bowl[mask]] @ d["V_bowl"])
    return softmax(z.astype(np.float64))


def nll(p: np.ndarray, y: np.ndarray) -> float:
    return -float(np.mean(np.log(np.maximum(p[np.arange(len(y)), y], 1e-12))))


def reliability(p: np.ndarray, y: np.ndarray, cls: str, bins=8) -> list:
    """Predicted vs observed frequency, in equal-count bins. Near-diagonal is what
    a simulator needs; a sharper but miscalibrated model is worse than useless."""
    c = CI[cls]
    pc, hit = p[:, c], (y == c).astype(np.float64)
    order = np.argsort(pc)
    rows = []
    for chunk in np.array_split(order, bins):
        if chunk.size:
            rows.append((float(pc[chunk].mean()), float(hit[chunk].mean()), chunk.size))
    return rows


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="random_by_match",
                    choices=["random_by_match", "chronological"])
    args = ap.parse_args()

    data = Data()
    tr, _, te = data.split(args.split)
    y = data.y[te]
    print(f"split={args.split}   test set: {te.sum()} balls, "
          f"{len(set(data.match[te].tolist()))} matches, "
          f"seasons {data.season[te].min()}-{data.season[te].max()}\n")
    prior = np.bincount(data.y[tr], minlength=N_CLASSES) / tr.sum()
    p_prior = np.tile(prior, (te.sum(), 1))

    print("scoring the classic engine on the same balls ...", flush=True)
    p_eng = engine_probs(data, te)
    p_mod = model_probs(data, te)

    print("\nnegative log-likelihood, all 9 classes (lower is better):")
    rows = [("prior (constant)", nll(p_prior, y)),
            ("classic engine", nll(p_eng, y)),
            ("learned backbone", nll(p_mod, y))]
    base = rows[0][1]
    for name, v in rows:
        print(f"  {name:<20} {v:.5f}   {100 * (base - v) / base:+5.2f}% vs prior")

    print("\nlegal deliveries only (the cricket, without the extras head):")
    legal_c = [CI[c] for c in ("0", "1", "2", "3", "4", "6", "Out")]
    lm = np.isin(y, legal_c)
    for name, p in (("prior (constant)", p_prior), ("classic engine", p_eng),
                    ("learned backbone", p_mod)):
        pl = p[lm][:, legal_c]
        pl = pl / pl.sum(axis=1, keepdims=True)
        remap = {c: i for i, c in enumerate(legal_c)}
        yl = np.array([remap[v] for v in y[lm]])
        print(f"  {name:<20} {nll(pl, yl):.5f}")

    era_breakdown(data, te, y, p_eng, p_mod, p_prior)

    for cls, label in (("Out", "P(wicket)"), ("6", "P(six)")):
        print(f"\nreliability, {label} -- test set")
        print(f"  {'bin':>4} {'n':>7} {'engine pred':>12} {'model pred':>11} {'observed':>10}")
        re_ = reliability(p_eng, y, cls)
        rm_ = reliability(p_mod, y, cls)
        for i, ((pe, oe, n), (pm, om, _)) in enumerate(zip(re_, rm_)):
            print(f"  {i:>4} {n:>7} {100 * pe:>11.2f}% {100 * pm:>10.2f}% {100 * om:>9.2f}%")
        print(f"  {'':>4} {'':>7} {'':>12} {'(model bins are its own ordering)':>11}")


if __name__ == "__main__":
    main()
