# `ml/` — learned ball model, built in isolation

An experimental replacement for the base probability distribution in the ball engine,
learned from 1,243 IPL matches (~296k deliveries) instead of hand-tuned.

## The one rule

**Nothing under `src/`, `config/`, `tests/` or the repo-root `requirements.txt` is ever
modified by anything in here.** The classic game keeps running exactly as it does today:

```
python src/server.py                  # classic, port 8000  — unchanged
python ml/run_server_with_model.py    # learned, port 8001  — same checkout
```

This package imports from `src/` read-only. It reuses `calculate_single_ball`,
`MatchState`, `apply_conditions`, `apply_modes` and `apply_roles` as they are.

The shadow launcher works without a source edit because [src/server.py](../src/server.py)
does `from src.engine.simulator import calculate_single_ball` — a from-import resolved at
import time. Patching `src.engine.simulator.calculate_single_ball` *before* importing
`src.server` swaps the engine cleanly.

## Layout

```
ml/
  etl/replay.py         shared sequential Cricsheet replay  (the single source of truth
                        for eligibility filtering and derived per-ball match state)
  etl/build_table.py    replay -> ball_table.npz feature matrix
  harness/reference.py  real-data target statistics
  harness/simulate.py   N-innings sweep against an injectable ball_fn
  harness/compare.py    side-by-side report vs the real distributions
  train/                logit backbone (numpy), MLP residual head (torch), export
  runtime/              numpy forward pass + the engine adapter
  artifacts/            generated data and models (gitignored)
```

## Setup

```
py -m venv ml/.venv
ml/.venv/Scripts/python -m pip install -r ml/requirements.txt
```

Run everything from the repo root so the `src.*` and `ml.*` imports resolve:

```
ml/.venv/Scripts/python -m ml.harness.run_baseline
```

## Order of work

0. Harness against the **current** engine — the scoreboard. No model.
1. ETL — the ball table.
2a. Logit backbone (numpy). 2b. MLP residual head (torch, GPU).
3. Adapter + shadow server.
4. Variance calibration.

Phase 0 comes first on purpose: without a scoreboard, no change to the engine —
hand-tuned or learned — is measurable.
