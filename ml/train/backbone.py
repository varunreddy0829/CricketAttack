"""Phase 2a: the wide half -- a hierarchical multinomial logit, fitted in numpy.

    python -m ml.train.backbone [--iters 1500]

The model:

    logits[n, c] = alpha[c]
                 + (E_bat [bat_idx[n]]  @ V_bat )[c]
                 + (E_bowl[bowl_idx[n]] @ V_bowl)[c]
                 + (X[n] @ B)[c]

    E_bat  = A_bat  @ W_bat  + D_bat        # (P, rank)
    E_bowl = A_bowl @ W_bowl + D_bowl

`A` are the fixed observable anchors (career stats + playstyle grids from
players_historical.json); `W` projects them into a rank-4 player space; `D` is a
small L2-shrunk learned correction per player. That decomposition is what makes
cold start work -- a player never seen in training still gets `A @ W`, and the long
tail of sub-100-ball players shrinks toward it rather than fitting noise.

Why a logit at all: the classic engine already *is* one. Multiplying weights is
adding in log space and "normalise to 1000" is a softmax, so six stages of
multiplication then normalise is a softmax over a sum of log-multipliers. This
fits those coefficients instead of guessing them, and adds terms the engine has no
way to express.

~7,000 parameters over 283,000 rows. One full-batch gradient is a single matmul,
so this trains in under two minutes on CPU with no torch dependency.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from ml.etl.schema import CLASSES, ENTITY_RANK, N_CLASSES, SCHEMA_HASH

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
TABLE_PATH = os.path.join(ARTIFACTS, "ball_table.npz")
MODEL_PATH = os.path.join(ARTIFACTS, "backbone.npz")

# --- how the data is split -------------------------------------------------
#
# DEFAULT: random, BY MATCH, across all seasons.
#
# The split must be at match level, never ball level. Six balls in an over share
# batter, bowler, over, pitch and day-state, so a random ball-level split puts most
# of every innings in train and tests on near-duplicates -- held-out log-loss looks
# excellent and means nothing. Grouping by match removes that entirely.
#
# It must span all seasons because the game has moved a long way: 2008-2015 ran
# 7.50 rpo with 4.4% sixes, 2023-2026 runs 9.01 rpo with 7.6% (see
# ml/etl/season_shift.py). A chronological train<=2023 / test 2025-26 split fits a
# game that no longer exists and then grades it on one it never saw. Every era now
# appears in train, val and test.
#
# `chronological` remains available: it answers a different and still useful
# question -- can the model extrapolate FORWARD into a season it has never seen?
# Expect it to score worse, because that is genuinely harder.
SPLIT_MODE = "random_by_match"
VAL_FRAC, TEST_FRAC = 0.10, 0.15
SPLIT_SEED = 17

TRAIN_MAX_SEASON = 2023     # chronological mode only
VAL_SEASON = 2024

# Recency weighting matters more now that old cricket is in the training set:
# without it the model averages 2010 and 2026 and plays like neither. At a 2-season
# half-life, 2026 carries full weight, 2023 about a third, and 2012 effectively
# nothing for the shared coefficients -- while those old matches still contribute
# to the player effects of anyone who only played then.
RECENCY_HALF_LIFE = 2.0
L2_DELTA = 30.0             # shrinkage on the per-player corrections
L2_CONTEXT = 1.0
L2_PROJ = 1.0


# --- data ------------------------------------------------------------------

class Data:
    def __init__(self, path: str = TABLE_PATH):
        d = np.load(path, allow_pickle=True)
        got = str(d["schema_hash"])
        if got != SCHEMA_HASH:
            raise SystemExit(
                f"schema mismatch: table was built with {got}, code is {SCHEMA_HASH}.\n"
                f"Rebuild with: python -m ml.etl.build_table"
            )
        self.X = d["X"].astype(np.float32)
        self.y = d["y"].astype(np.int64)
        self.bat = d["bat_idx"].astype(np.int64)
        self.bowl = d["bowl_idx"].astype(np.int64)
        self.season = d["season"].astype(np.int32)
        self.match = d["match_idx"].astype(np.int64)
        self.A_bat = d["bat_anchors"].astype(np.float32)
        self.A_bowl = d["bowl_anchors"].astype(np.float32)
        self.names = d["player_names"]
        self.n_players = self.A_bat.shape[0]

    def split(self, mode: str = SPLIT_MODE, seed: int = SPLIT_SEED):
        """-> (train, val, test) boolean masks over balls."""
        if mode == "chronological":
            return (self.season <= TRAIN_MAX_SEASON,
                    self.season == VAL_SEASON,
                    self.season > VAL_SEASON)

        if mode != "random_by_match":
            raise ValueError(f"unknown split mode {mode!r}")

        # assign whole MATCHES, never individual balls -- balls within a match share
        # players, conditions and day-state, so a ball-level split is self-leaking
        ids = np.unique(self.match)
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(ids)
        n_val = int(round(VAL_FRAC * len(ids)))
        n_test = int(round(TEST_FRAC * len(ids)))
        val_ids = set(shuffled[:n_val].tolist())
        test_ids = set(shuffled[n_val:n_val + n_test].tolist())

        va = np.isin(self.match, list(val_ids))
        te = np.isin(self.match, list(test_ids))
        return ~(va | te), va, te

    def recency_weights(self, mask, half_life: float = RECENCY_HALF_LIFE) -> np.ndarray:
        """Exponential decay by season. Applied to the TRAINING rows only, so the
        model plays like recent cricket while still using two decades of matches to
        pin down player effects."""
        s = self.season[mask].astype(np.float32)
        return np.power(0.5, (self.season.max() - s) / half_life).astype(np.float32)


# --- model -----------------------------------------------------------------

class Backbone:
    def __init__(self, n_ctx: int, n_players: int, n_bat_anchor: int,
                 n_bowl_anchor: int, rank: int = ENTITY_RANK, seed: int = 0):
        rng = np.random.default_rng(seed)
        s = 0.01
        self.alpha = np.zeros(N_CLASSES, dtype=np.float32)
        self.B = (s * rng.standard_normal((n_ctx, N_CLASSES))).astype(np.float32)
        self.W_bat = (s * rng.standard_normal((n_bat_anchor, rank))).astype(np.float32)
        self.W_bowl = (s * rng.standard_normal((n_bowl_anchor, rank))).astype(np.float32)
        self.V_bat = (s * rng.standard_normal((rank, N_CLASSES))).astype(np.float32)
        self.V_bowl = (s * rng.standard_normal((rank, N_CLASSES))).astype(np.float32)
        self.D_bat = np.zeros((n_players, rank), dtype=np.float32)
        self.D_bowl = np.zeros((n_players, rank), dtype=np.float32)
        self.rank = rank

    @property
    def params(self):
        return ["alpha", "B", "W_bat", "W_bowl", "V_bat", "V_bowl", "D_bat", "D_bowl"]

    def embeddings(self, A_bat, A_bowl):
        return A_bat @ self.W_bat + self.D_bat, A_bowl @ self.W_bowl + self.D_bowl

    def logits(self, X, bat, bowl, A_bat, A_bowl):
        E_bat, E_bowl = self.embeddings(A_bat, A_bowl)
        return (
            self.alpha
            + X @ self.B
            + E_bat[bat] @ self.V_bat
            + E_bowl[bowl] @ self.V_bowl
        )


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=1, keepdims=True)
    return z


def _scatter(idx: np.ndarray, vals: np.ndarray, n: int) -> np.ndarray:
    """Sum `vals` rows into `n` bins by `idx`. bincount per column beats
    np.add.at by roughly an order of magnitude at this size."""
    out = np.zeros((n, vals.shape[1]), dtype=np.float64)
    for r in range(vals.shape[1]):
        out[:, r] = np.bincount(idx, weights=vals[:, r], minlength=n)
    return out.astype(np.float32)


def loss_and_grads(m: Backbone, X, y, bat, bowl, A_bat, A_bowl, w):
    n = X.shape[0]
    E_bat, E_bowl = m.embeddings(A_bat, A_bowl)
    eb, ew = E_bat[bat], E_bowl[bowl]

    z = m.alpha + X @ m.B + eb @ m.V_bat + ew @ m.V_bowl
    p = softmax(z)

    wn = w / w.sum()
    nll = -float(np.sum(wn * np.log(np.maximum(p[np.arange(n), y], 1e-12))))

    dz = p
    dz[np.arange(n), y] -= 1.0
    dz *= wn[:, None]

    g = {}
    g["alpha"] = dz.sum(axis=0).astype(np.float32)
    g["B"] = (X.T @ dz).astype(np.float32) + 2 * L2_CONTEXT / n * m.B
    g["V_bat"] = (eb.T @ dz).astype(np.float32)
    g["V_bowl"] = (ew.T @ dz).astype(np.float32)

    dE_bat = _scatter(bat, dz @ m.V_bat.T, m.D_bat.shape[0])
    dE_bowl = _scatter(bowl, dz @ m.V_bowl.T, m.D_bowl.shape[0])

    g["W_bat"] = (A_bat.T @ dE_bat).astype(np.float32) + 2 * L2_PROJ / n * m.W_bat
    g["W_bowl"] = (A_bowl.T @ dE_bowl).astype(np.float32) + 2 * L2_PROJ / n * m.W_bowl
    g["D_bat"] = dE_bat + 2 * L2_DELTA / n * m.D_bat
    g["D_bowl"] = dE_bowl + 2 * L2_DELTA / n * m.D_bowl
    return nll, g


def eval_nll(m: Backbone, X, y, bat, bowl, A_bat, A_bowl) -> float:
    p = softmax(m.logits(X, bat, bowl, A_bat, A_bowl))
    return -float(np.mean(np.log(np.maximum(p[np.arange(len(y)), y], 1e-12))))


def fit(data: Data, *, iters: int = 1500, lr: float = 0.05, seed: int = 0,
        split_mode: str = SPLIT_MODE, half_life: float = RECENCY_HALF_LIFE,
        verbose: bool = True) -> Backbone:
    tr, va, _ = data.split(split_mode)
    Xtr, ytr, btr, wtr = data.X[tr], data.y[tr], data.bat[tr], data.bowl[tr]
    Xva, yva, bva, wva = data.X[va], data.y[va], data.bat[va], data.bowl[va]
    weights = data.recency_weights(tr, half_life)

    m = Backbone(data.X.shape[1], data.n_players,
                 data.A_bat.shape[1], data.A_bowl.shape[1], seed=seed)

    # Adam
    mom = {k: np.zeros_like(getattr(m, k)) for k in m.params}
    vel = {k: np.zeros_like(getattr(m, k)) for k in m.params}
    b1, b2, eps = 0.9, 0.999, 1e-8

    best_val, best_state, stale = float("inf"), None, 0
    t0 = time.time()
    for it in range(1, iters + 1):
        nll, g = loss_and_grads(m, Xtr, ytr, btr, wtr, data.A_bat, data.A_bowl, weights)
        for k in m.params:
            gk = g[k]
            mom[k] = b1 * mom[k] + (1 - b1) * gk
            vel[k] = b2 * vel[k] + (1 - b2) * gk * gk
            mh = mom[k] / (1 - b1 ** it)
            vh = vel[k] / (1 - b2 ** it)
            setattr(m, k, getattr(m, k) - lr * mh / (np.sqrt(vh) + eps))

        if it % 50 == 0 or it == iters:
            v = eval_nll(m, Xva, yva, bva, wva, data.A_bat, data.A_bowl)
            if verbose:
                print(f"    iter {it:>5}  train {nll:.5f}  val {v:.5f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
            if v < best_val - 1e-5:
                best_val, stale = v, 0
                best_state = {k: getattr(m, k).copy() for k in m.params}
            else:
                stale += 1
                if stale >= 4:
                    if verbose:
                        print(f"    early stop at iter {it} (val {best_val:.5f})")
                    break

    if best_state:
        for k, v in best_state.items():
            setattr(m, k, v)
    return m


def save(m: Backbone, data: Data, path: str = MODEL_PATH) -> None:
    E_bat, E_bowl = m.embeddings(data.A_bat, data.A_bowl)
    np.savez_compressed(
        path,
        alpha=m.alpha, B=m.B,
        W_bat=m.W_bat, W_bowl=m.W_bowl,
        V_bat=m.V_bat, V_bowl=m.V_bowl,
        D_bat=m.D_bat, D_bowl=m.D_bowl,
        E_bat=E_bat, E_bowl=E_bowl,           # precomputed for the runtime
        player_names=data.names,
        schema_hash=np.asarray(SCHEMA_HASH),
        classes=np.asarray(CLASSES),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default=SPLIT_MODE,
                    choices=["random_by_match", "chronological"])
    ap.add_argument("--half-life", type=float, default=RECENCY_HALF_LIFE,
                    help="recency half-life in seasons; lower = more modern")
    args = ap.parse_args()

    data = Data()
    tr, va, te = data.split(args.split)
    print(f"rows {data.X.shape[0]}  features {data.X.shape[1]}  players {data.n_players}")
    print(f"split mode: {args.split}")
    print(f"  train {tr.sum():>7}   val {va.sum():>6}   test {te.sum():>6}")
    if args.split == "random_by_match":
        print(f"  matches: {len(np.unique(data.match[tr]))} / "
              f"{len(np.unique(data.match[va]))} / {len(np.unique(data.match[te]))}"
              f"   (whole matches, never split mid-innings)")
        print(f"  test seasons span {data.season[te].min()}-{data.season[te].max()}")

    n_params = sum(getattr(Backbone(data.X.shape[1], data.n_players,
                                    data.A_bat.shape[1], data.A_bowl.shape[1]), k).size
                   for k in ["alpha", "B", "W_bat", "W_bowl", "V_bat", "V_bowl",
                             "D_bat", "D_bowl"])
    print(f"parameters {n_params}\n")

    print(f"recency half-life {args.half_life} seasons\n")
    m = fit(data, iters=args.iters, lr=args.lr, seed=args.seed,
            split_mode=args.split, half_life=args.half_life)

    print("\nheld-out negative log-likelihood (lower is better):")
    for name, mask in (("train", tr), ("val", va), ("test", te)):
        v = eval_nll(m, data.X[mask], data.y[mask], data.bat[mask], data.bowl[mask],
                     data.A_bat, data.A_bowl)
        print(f"  {name:<6} {v:.5f}")

    # constant-rate floor: the best you can do knowing only the class frequencies
    prior = np.bincount(data.y[tr], minlength=N_CLASSES) / tr.sum()
    floor = -float(np.mean(np.log(np.maximum(prior[data.y[te]], 1e-12))))
    print(f"  {'(prior)':<6} {floor:.5f}   <- constant predictor on test")

    # Broken out by era, because recency weighting deliberately trades accuracy on
    # 2010 cricket for accuracy on 2026 cricket. The recent rows are the ones that
    # matter for a game meant to feel like the IPL people actually watch.
    print("\ntest NLL by era (recency weighting favours the recent rows on purpose):")
    for label, lo, hi in (("2008-2015", 2007, 2015), ("2016-2022", 2016, 2022),
                          ("2023-2026", 2023, 2026)):
        em = te & (data.season >= lo) & (data.season <= hi)
        if em.sum() < 200:
            continue
        v = eval_nll(m, data.X[em], data.y[em], data.bat[em], data.bowl[em],
                     data.A_bat, data.A_bowl)
        pf = -float(np.mean(np.log(np.maximum(prior[data.y[em]], 1e-12))))
        print(f"  {label:<10} n={em.sum():>6}   model {v:.5f}   prior {pf:.5f}"
              f"   {100 * (pf - v) / pf:+5.2f}%")

    save(m, data)
    print(f"\nwrote {MODEL_PATH}")


if __name__ == "__main__":
    main()
