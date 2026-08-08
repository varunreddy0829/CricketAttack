# Repo docs

Plain-language explanations of how different parts of this project work — written
for understanding, not as API reference (the code comments handle that). Numbers
quoted here are measured, not estimated; re-run the referenced script if you want
to verify one yourself.

## Categories

- [model.md](model.md) — the learned ball-outcome model in `ml/`: what it is, what
  it looks at, how it was built and trained, and how we know it's any good.
- [eras.md](eras.md) — why the game splits into eras, what's scoped per era, and
  how OVRs are measured through the model instead of computed from formulas.

More categories will land here as they're written (data pipeline, the classic
engine, the server/game-state machine, etc).
