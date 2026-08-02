"""Play the game with the learned model. Classic stays on :8000, untouched.

    ml/.venv/Scripts/python -m ml.run_server_with_model            # :8001
    ml/.venv/Scripts/python -m ml.run_server_with_model --port 9000

`src/server.py:34` does

    from src.engine.simulator import calculate_single_ball, EXTRAS_PROB, WIDE_SHARE_OF_EXTRAS

which resolves those names at import time, so patching the simulator module BEFORE
importing the server swaps the engine cleanly. Run both at once and compare.

Extras (EXTRAS_PROB, WIDE_SHARE_OF_EXTRAS) are no longer patched here -- they were
measured from the same real-data pipeline this whole package is built on, and are
now the values `src/engine/simulator.py` ships with directly, so BOTH engines use
them without any override.

Everything the two players control still works exactly as authored: the batting and
bowling roles, the playstyle-grid bonus and the gambit cards are all applied on top
of the model, untouched. The wicket cascade is the one exception -- it's OFF here
(see the `cascade=False` note below for why); every other layer of player control
is unchanged. Only the base distribution changes.
"""

from __future__ import annotations

import argparse
import json
import os

ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
CALIBRATION_PATH = os.path.join(ARTIFACTS, "model_calibration.json")


def load_calibration() -> dict:
    defaults = {"day_sigma": 0.225, "calibration": 1.2437, "out_calibration": 1.40}
    try:
        with open(CALIBRATION_PATH, "r", encoding="utf-8") as fh:
            defaults.update(json.load(fh))
    except OSError:
        pass
    return defaults


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--no-day-factor", action="store_true")
    args = ap.parse_args()

    cal = load_calibration()
    day_sigma = 0.0 if args.no_day_factor else cal["day_sigma"]

    from ml.runtime import server_ctx
    from ml.runtime.adapter import make_ball_fn
    from ml.runtime.model import OutcomeModel

    model = OutcomeModel.load()
    inner = make_ball_fn(
        model.base_provider(),
        player_stages=False,          # the model already knows who is batting
        # OFF here (unlike the classic engine): the model already sees a fresh
        # batter directly -- striker_balls/is_set/partnership_balls all read 0 on
        # his first ball -- and learned the real effect from history. Stacking the
        # classic engine's hand-tuned 0.45 damping on top double-suppresses it; see
        # ml/harness/run_model.py's docstring for the measured comparison.
        cascade=False,
        calibration=cal["calibration"],
        out_calibration=cal["out_calibration"],
    )

    import src.engine.simulator as sim

    def patched(striker, bowler, league_avg, context=None):
        # GAME is read live: it doesn't exist yet at patch time
        game = getattr(_server[0], "GAME", None) if _server else None
        ctx = context or {}
        if game:
            ctx = server_ctx.enrich(ctx, striker, bowler, game, day_sigma=day_sigma)
        return inner(striker, bowler, league_avg, ctx)

    _server: list = []
    sim.calculate_single_ball = patched          # BEFORE the server imports it

    import src.server as server
    _server.append(server)

    print("=" * 62)
    print("  LEARNED MODEL  (src/ untouched -- classic still runs on :8000)")
    print(f"  calibration {cal['calibration']:.4f} | "
          f"out {cal['out_calibration']:.4f} | day sigma {day_sigma:.3f}")
    print(f"  http://localhost:{args.port}")
    print("=" * 62)
    server.app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
