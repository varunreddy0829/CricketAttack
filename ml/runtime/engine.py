"""Resolve which ball engine the game runs on.

`src/server.py` calls `resolve_engine()` once at import and uses whatever comes
back. The learned model is the default; the classic hand-tuned engine is the
fallback if the model can't be loaded for any reason.

The fallback is not decoration. The server must boot even if `backbone.npz` is
missing, corrupt, or built against a different feature schema -- a game that
refuses to start is worse than one running the older engine, and the schema hash
check in OutcomeModel is specifically designed to fail loudly rather than serve
silently-wrong probabilities. Whichever path is taken gets printed at startup so
it's never a mystery which engine is live.
"""

from __future__ import annotations

import json
import os

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "artifacts")
CALIBRATION_PATH = os.path.join(ARTIFACTS, "model_calibration.json")

# Fitted against real 2023-26 cricket by ml/harness/calibrate_variance.py. These
# are the fallbacks if the file is missing; the file is the source of truth.
DEFAULT_CALIBRATION = {
    "calibration": 1.075,
    "out_calibration": 1.05,
    "day_sigma": 0.1781,
}


def load_calibration(era_id: str | None = None) -> dict:
    """The three fitted constants for one engine. `era_id` selects that era's
    file; each era is calibrated against its OWN real innings, which differ a
    lot -- 150 runs an innings in 2008-2013 against 181 in 2023-2026."""
    path = (CALIBRATION_PATH if era_id in (None, "all_time")
            else os.path.join(ARTIFACTS, "eras", era_id, "model_calibration.json"))
    cal = dict(DEFAULT_CALIBRATION)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cal.update(json.load(fh))
    except (OSError, ValueError):
        pass
    return cal


def resolve_engine(era_id: str | None = None):
    """-> (ball_fn, enrich_fn | None, description)

    `ball_fn` has exactly `calculate_single_ball`'s signature.
    `enrich_fn(ctx, striker, bowler, game) -> ctx` adds the match state the model
    needs; None on the classic path, which ignores those fields anyway.

    `era_id` selects that era's model, calibration and venue rates. None gives
    the career-wide model.
    """
    from src.engine.simulator import calculate_single_ball as classic

    try:
        from ml.runtime import server_ctx
        from ml.runtime.adapter import make_ball_fn
        from ml.runtime.model import OutcomeModel

        model = OutcomeModel.load(era_id=era_id)
        cal = load_calibration(era_id)

        ball_fn = make_ball_fn(
            model.base_provider(),
            player_stages=False,   # the model already knows who is batting
            cascade=False,         # the model already sees a new batter directly
            new_roles=True,        # ml/runtime/roles.py -- explicit paired transfers
            calibration=cal["calibration"],
            out_calibration=cal["out_calibration"],
        )
        sigma = cal["day_sigma"]

        def enrich(ctx, striker, bowler, game):
            return server_ctx.enrich(ctx, striker, bowler, game, day_sigma=sigma,
                                     era_id=era_id)

        desc = (f"learned model (calibration {cal['calibration']:.3f}, "
                f"out {cal['out_calibration']:.3f}, day sigma {sigma:.3f})")
        return ball_fn, enrich, desc

    except Exception as exc:   # noqa: BLE001 -- any failure must still boot the game
        return classic, None, f"CLASSIC fallback -- model unavailable: {exc}"
