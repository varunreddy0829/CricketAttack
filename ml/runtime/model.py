"""Runtime forward pass: a few matmuls, numpy only, no torch.

Loads the exported arrays and turns live game state into the engine's 8-key
1000-sum weight dict, plus per-ball extras probabilities.

The schema hash is checked at load. A model trained against a different feature
layout fails loudly here rather than silently mis-ordering columns -- which would
produce plausible-looking nonsense that no test would catch.
"""

from __future__ import annotations

import math
import os

import numpy as np

from ml.etl.schema import CLASSES, N_CONTEXT, SCHEMA_HASH
from ml.runtime import features as F


def _features():
    return F

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
DEFAULT_MODEL = os.path.join(ARTIFACTS, "backbone.npz")

ENGINE_KEYS = ("0", "1", "2", "3", "4", "5", "6", "Out")
# '5' is not modelled and never sampled -- the key stays in the dict only because
# the engine's downstream stages (roles.py, the scorecard) expect exactly these 8
# keys to exist; a weight of 0 makes it structurally impossible for the model path
# to ever produce a '5' rather than pretending to know its rate.
FIVE_WEIGHT = 0.0

_LEGAL = ("0", "1", "2", "3", "4", "6", "Out")

# Direction in logit space that the per-innings day factor pushes along: a "road"
# means more boundaries and fewer dots/wickets, a "minefield" the reverse. Scaled
# by a single sigma fitted in Phase 4 -- see ml/harness/calibrate_variance.py.
REPLACEMENT_PCT = 0.15   # bottom 15% of a pool defines replacement level

# How much evidence before a player's own rating is half-trusted, and what the
# rest of him falls back to. See set_shrinkage for why the target must be
# REPLACEMENT and not zero: zero is better than 0% of batters but 99% of bowlers,
# so the same rule inverts on one side.
SHRINK_BALLS = 500
SHRINK_TARGET = "replacement"

DAY_AXIS = {"0": -0.6, "1": -0.1, "2": 0.1, "3": 0.1,
            "4": 0.7, "6": 0.9, "Out": -0.5, "wide": 0.0, "noball": 0.0}


class OutcomeModel:
    def __init__(self, d):
        got = str(d["schema_hash"])
        if got != SCHEMA_HASH:
            raise RuntimeError(
                f"model/schema mismatch: model built with {got}, code is {SCHEMA_HASH}. "
                f"Retrain with: python -m ml.train.backbone"
            )
        self.alpha = d["alpha"].astype(np.float64)
        self.B = d["B"].astype(np.float64)
        self.V_bat = d["V_bat"].astype(np.float64)
        self.V_bowl = d["V_bowl"].astype(np.float64)
        self.E_bat = d["E_bat"].astype(np.float64)
        self.E_bowl = d["E_bowl"].astype(np.float64)
        # the shared projections, kept so a player this model never saw can still
        # be given a real effect from his own observables -- see cold_effect
        self.W_bat = d["W_bat"].astype(np.float64)
        self.W_bowl = d["W_bowl"].astype(np.float64)
        self._cold: dict = {}
        names = [str(n) for n in d["player_names"]]
        self.idx = {n: i for i, n in enumerate(names)}
        self.classes = [str(c) for c in d["classes"]] if "classes" in d else list(CLASSES)
        self.ci = {c: i for i, c in enumerate(self.classes)}
        self.axis = np.array([DAY_AXIS.get(c, 0.0) for c in self.classes])
        self._row = np.zeros(N_CONTEXT, dtype=np.float32)

    @classmethod
    def load(cls, path: str | None = None, era_id: str | None = None) -> "OutcomeModel":
        """Load a trained model. `era_id` selects that era's artifact; omitting
        both gives the career-wide model."""
        if path is None:
            path = (DEFAULT_MODEL if era_id in (None, "all_time")
                    else os.path.join(ARTIFACTS, "eras", era_id, "backbone.npz"))
        return cls(np.load(path, allow_pickle=True))

    # --- core ---------------------------------------------------------------

    def cold_effect(self, record: dict, side: str) -> np.ndarray:
        """The player effect for someone this model has NEVER seen: A . W.

        Index 0 is a zero row, so an unrecognised name otherwise contributes
        NOTHING and every stranger plays as an identical league-average player.
        That is fine for one surprise auction pick and useless for a pool built
        of them -- the multiverse era tags names by era ("V Kohli (14-22)"), so
        every single player would be a stranger.

        Projecting the player's own observables through the shared W recovers a
        real, differentiated effect without the learned per-player correction D.
        That is exactly the cold-start the low-rank design exists for: E = A.W + D,
        and D is the small part.
        """
        F = _features()
        if side == "bat":
            return F.bat_anchor(record).astype(np.float64) @ self.W_bat
        return F.bowl_anchor(record).astype(np.float64) @ self.W_bowl

    def set_shrinkage(self, records: list[dict], k: float) -> None:
        """Regress every player's effect toward league average by how little we
        saw of him:  effect x balls / (balls + k).

        A rating built on 300 balls is a guess with wide error bars, and the model
        has no way to say so -- it reports the same confident effect for a man with
        300 balls as for one with 3,000. Measured on the 2014-2022 grid, that put
        Tilak Varma (303 balls, one season) 12th and Hetmyer (546) 4th, above
        Kohli. Shrinking by sample size moves them to 60th and 39th while Kohli
        rises to 4th, and it does so without inventing anything: it only ever pulls
        a player TOWARD the mean, never past his measured rating.

        `k` is in balls -- the evidence required before a player is half-trusted.
        Zero disables it. This is deliberately separate from the longevity layer:
        that one moves `Out` after the fact, this one damps the whole effect
        including scoring rate, which is the part the transfer cannot reach.

        WHAT it shrinks toward matters more than how hard, and two wrong answers
        were measured before the right one:

          toward ZERO        10/16 outside tolerance. Zero is not the average
                             player, it is the model's blank slate, so this drags
                             the whole league down -- median innings 162 -> 133,
                             innings over 200 from 9.0% to 0.6%.
          toward the MEAN    2/16. Better, but it RESCUES TAILENDERS: Chahal has
                             86 career balls and Bumrah 66, so they shrink hardest
                             and end up batting like average players. All-out rate
                             collapsed from a real 8.1% to 3.8%. Their low ratings
                             were right; the small sample was not the reason.
          toward A.W         the answer, below.

        The model is E = A.W + D: `A.W` projects a player from HIS OWN observable
        stats, and `D` is the learned per-player correction on top. `D` is the part
        that overfits a short career -- `A.W` is already an honest read of whoever
        he is. So only `D` is shrunk. A tailender keeps the bad projection his own
        numbers earn him, while a 300-ball batter loses the flattering correction
        the model could not have learned reliably.
        """
        self.shrink_k = float(k)
        self._shrink = {}
        self._anchor = {}
        if not k:
            return
        for r in records:
            nb = (r.get("batting") or {}).get("balls", 0)
            nw = (r.get("bowling") or {}).get("legal_balls", 0)
            self._shrink[r["name"]] = (nb / (nb + k), nw / (nw + k))
            self._anchor[r["name"]] = (self.cold_effect(r, "bat"),
                                       self.cold_effect(r, "bowl"))
        # REPLACEMENT LEVEL, per side: the ball-weighted mean effect of the
        # bottom REPLACEMENT_PCT of the pool by quality. This is the only target
        # that means the same thing on both sides.
        #
        # `zero` does not. Measured on 2014-2022, a zero effect is better than 0%
        # of batters but 99% of BOWLERS -- the two effect distributions sit on
        # opposite sides of the origin, because a blank-slate player concedes no
        # boundaries (great for a bowler) and hits none (terrible for a batter).
        # So the identical rule promoted fringe bowlers while demoting fringe
        # batters: Badree (234 balls) went 33rd -> 5th while Chahal (2807 balls,
        # 159 wickets) went 60th -> 109th.
        # What "good" MEANS, per side. A batter wants runs -- 1s, 2s, 3s, 4s, 6s
        # -- and does not want dots or dismissals. A bowler wants exactly the
        # reverse: dots and wickets, not runs. The two roles sit on opposite sides
        # of the origin on both axes, which is why a single scalar shrink rule
        # inverts on one of them.
        scoring = [(c, int(c)) for c in ("1", "2", "3", "4", "6") if c in self.ci]

        def _quality(e, side):
            runs = sum(v * e[self.ci[c]] for c, v in scoring)
            stop = e[self.ci["0"]] + e[self.ci["Out"]]
            return (runs - stop) if side == "bat" else (stop - runs)

        def _raw(r, side):
            """This player's effect BEFORE shrinkage -- learned if the model has
            seen him, projected from his own stats if not.

            The multiverse tags every name ("V Kohli (The Shift)"), so none of
            them is in the index and an index-only lookup found NOBODY: the
            replacement level came back None and shrinkage silently did nothing
            in the one mode that most needs it. It rated 349-ball Finn Allen at
            99 and 32-wicket Faulkner at 96.
            """
            i = self.idx.get(r["name"])
            if i is not None:
                return self.E_bat[i] if side == "bat" else self.E_bowl[i]
            return self.cold_effect(r, side)

        def _replacement(recs, side, V, vol):
            scored = []
            for r in recs:
                n = vol(r)
                if n <= 0:
                    continue
                e = _raw(r, side) @ V
                scored.append((_quality(e, side), n, r["name"]))
            if not scored:
                return None
            scored.sort()
            cut = max(1, int(REPLACEMENT_PCT * len(scored)))
            tot = sum(n for _, n, _ in scored[:cut]) or 1.0
            acc = np.zeros_like(self.E_bat[0] if side == "bat" else self.E_bowl[0])
            by_name = {r["name"]: r for r in recs}
            for _, n, nm in scored[:cut]:
                acc += n * _raw(by_name[nm], side)
            return acc / tot

        self._repl_bat = _replacement(
            records, "bat", self.V_bat,
            lambda r: (r.get("batting") or {}).get("balls", 0))
        self._repl_bowl = _replacement(
            records, "bowl", self.V_bowl,
            lambda r: (r.get("bowling") or {}).get("legal_balls", 0))

        # the pool mean, ball-weighted, kept so SHRINK_TARGET can select it
        wb = ww = 0.0
        mb = np.zeros_like(self.E_bat[0])
        mw = np.zeros_like(self.E_bowl[0])
        for r in records:
            i = self.idx.get(r["name"])
            if i is None:
                continue
            nb = (r.get("batting") or {}).get("balls", 0)
            nw = (r.get("bowling") or {}).get("legal_balls", 0)
            mb += nb * self.E_bat[i]; wb += nb
            mw += nw * self.E_bowl[i]; ww += nw
        self._mean_bat = mb / wb if wb else mb
        self._mean_bowl = mw / ww if ww else mw

    def _shrink_factor(self, name: str, side: str) -> float:
        if not getattr(self, "shrink_k", 0.0):
            return 1.0
        f = self._shrink.get(name)
        if f is None:
            return 0.0        # never seen at all -- trust nothing, use the mean
        return f[0] if side == "bat" else f[1]

    def effect(self, name: str, side: str, record: dict | None = None) -> np.ndarray:
        """Player effect by name, falling back to the anchor projection."""
        i = self.idx.get(name)
        if i is not None:
            e = self.E_bat[i] if side == "bat" else self.E_bowl[i]
        elif record is None:
            return self.E_bat[0] if side == "bat" else self.E_bowl[0]
        else:
            key = (side, name)
            e = self._cold.get(key)
            if e is None:
                e = self._cold[key] = self.cold_effect(record, side)
        f = self._shrink_factor(name, side)
        if f == 1.0:
            return e
        t = getattr(self, "shrink_target", "anchor")
        if t == "replacement":
            base = self._repl_bat if side == "bat" else self._repl_bowl
            if base is None:
                return e
        elif t == "zero":
            base = 0.0 * e
        elif t == "mean":
            base = self._mean_bat if side == "bat" else self._mean_bowl
        else:
            a = self._anchor.get(name)
            if a is None:
                return e                   # no record to project from; leave it
            base = a[0] if side == "bat" else a[1]
        return base + (e - base) * f

    def logits(self, striker_name: str, bowler_name: str, row: np.ndarray,
               z: float = 0.0, bat_record: dict | None = None,
               bowl_record: dict | None = None) -> np.ndarray:
        out = (self.alpha
               + row @ self.B
               + self.effect(striker_name, "bat", bat_record) @ self.V_bat
               + self.effect(bowler_name, "bowl", bowl_record) @ self.V_bowl)
        if z:
            out = out + z * self.axis
        return out

    def probs(self, striker_name, bowler_name, row, z=0.0,
              bat_record=None, bowl_record=None) -> np.ndarray:
        z_ = self.logits(striker_name, bowler_name, row, z,
                         bat_record=bat_record, bowl_record=bowl_record)
        z_ -= z_.max()
        e = np.exp(z_)
        return e / e.sum()

    def predict(self, striker, bowler, ctx: dict) -> tuple[dict, float, float]:
        """-> (8-key weights summing to 1000.0, p_wide, p_noball)"""
        row = self._row
        row.fill(0.0)
        F.build_row(
            row,
            over=ctx.get("over_num", 0) or 0,
            ball_in_over=ctx.get("ball_in_over", 1),
            wickets=ctx.get("wickets", 0),
            balls_remaining=ctx.get("balls_remaining", 120),
            innings_no=ctx.get("innings_no", 1),
            score=ctx.get("score", 0),
            target=ctx.get("target"),
            striker_balls=ctx.get("striker_balls", 0),
            striker_position=ctx.get("striker_position", 1),
            bowler_balls=ctx.get("bowler_balls", 0),
            over_in_spell=ctx.get("over_in_spell", 1),
            bat_career_balls=ctx.get("bat_career_balls", getattr(striker, "career_balls", 0)),
            bowl_career_balls=ctx.get("bowl_career_balls", getattr(bowler, "legal_balls", 0)),
            ns_sr=ctx.get("ns_sr", 120.0),
            venue_rpb=ctx.get("venue_rpb", 1.358),
            venue_wpb=ctx.get("venue_wpb", 0.0493),
            venue_bdry_share=ctx.get("venue_bdry_share", 0.589),
            venue_type_edge=ctx.get("venue_type_edge", 0.0),
            edges=ctx.get("edges"),
        )
        p = self.probs(striker.name, bowler.name, row, ctx.get("day_factor", 0.0),
                       bat_record=ctx.get("bat_record"),
                       bowl_record=ctx.get("bowl_record"))

        p_wide = float(p[self.ci["wide"]])
        p_nb = float(p[self.ci["noball"]])

        legal_total = sum(float(p[self.ci[c]]) for c in _LEGAL) or 1.0
        budget = 1000.0 - FIVE_WEIGHT
        w = {c: float(p[self.ci[c]]) / legal_total * budget for c in _LEGAL}
        w["5"] = FIVE_WEIGHT
        return {k: w[k] for k in ENGINE_KEYS}, p_wide, p_nb

    # --- adapter plumbing ----------------------------------------------------

    def base_provider(self):
        def provider(striker, bowler, ctx):
            return self.predict(striker, bowler, ctx)[0]
        return provider


def wicket_kind_table():
    """p(kind | phase) -- measured, not modelled. `Out` is ~4.9% of balls and
    splitting it five ways puts classes at 0.2-2.5%, which spends real signal on
    flavour text. Sampled after Out is drawn."""
    return {
        "pp":    [("caught", 0.60), ("bowled", 0.21), ("lbw", 0.07),
                  ("run out", 0.05), ("caught and bowled", 0.03), ("stumped", 0.04)],
        "mid":   [("caught", 0.62), ("bowled", 0.16), ("lbw", 0.06),
                  ("run out", 0.05), ("caught and bowled", 0.03), ("stumped", 0.08)],
        "death": [("caught", 0.70), ("bowled", 0.17), ("lbw", 0.05),
                  ("run out", 0.05), ("caught and bowled", 0.02), ("stumped", 0.01)],
    }
