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

    def effect(self, name: str, side: str, record: dict | None = None) -> np.ndarray:
        """Player effect by name, falling back to the anchor projection."""
        i = self.idx.get(name)
        if i is not None:
            return self.E_bat[i] if side == "bat" else self.E_bowl[i]
        if record is None:
            return self.E_bat[0] if side == "bat" else self.E_bowl[0]
        key = (side, name)
        hit = self._cold.get(key)
        if hit is None:
            hit = self._cold[key] = self.cold_effect(record, side)
        return hit

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
