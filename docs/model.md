# The learned ball model — a mental model

**Status: live.** The server runs on learned models — one per playable era —
with the original hand-tuned engine kept as the All-Time mode. Everything
supporting them lives under `ml/`; `src/` holds the game around it.

Each era's model reproduces its own cricket closely:

| era | real innings | model | engine |
|---|---|---|---|
| 2008–2013 | 150.9 | **149.8** | learned |
| 2014–2022 | 160.9 | **160.2** | learned |
| 2023–2026 | 181.5 | **180.5** | learned |
| All-Time | — | — | classic hand-tuned |

See [eras.md](eras.md) for why the split exists and how OVRs are derived.
This document explains the model itself, which is the same design in every era.

## The one-sentence version

For any given ball, the model answers *"what happens next?"* with one list of nine
percentages that add up to 100 — the same job the classic engine's six hand-tuned
stages do today, except the numbers come from what actually happened in 290,611
real IPL deliveries instead of from someone's best guess.

```
dot 28% · 1 run 31% · 2 runs 7% · 3 runs 1% · 4 runs 13%
6 runs 9% · WICKET 8% · wide 2% · no-ball 1%
```

The game rolls a die against that list, exactly like it always has. Only where the
list comes from is different.

## Where it sits in the pipeline

```
                       ┌─ MODEL: the 9 "gold probs"
Stage 0 baseline  ─────┤
Stage 1 OVR       ─────┤   replaces these five stages
Stage 2 SR/eco    ─────┤
Stage 3 wicket    ─────┤
Stage 3.5 pitch/phase ─┘
────────────────────────────────────────────────────────
Stage 3.5 gambits          ← unchanged (one-shot cards)
Role play                  ← REWRITTEN, ml/runtime/roles.py
Stage 5 skill grid         ← REMOVED (already a model input)
Stage 6 wicket cascade     ← REMOVED (already a model input)
          sample the outcome
```

The model decides "how good is this matchup, here, right now." Everything a
player *chooses* still sits on top of it — that layer is what makes this a game
rather than a simulation you watch. Player choices can't be model inputs, because
Cricsheet records what happened and never what anyone was **trying** to do.

Two of the old hand-tuned layers were **removed rather than kept**, both for the
same reason: the model already has that information as a direct input, so applying
the old adjustment on top counted the same thing twice.

- **Skill grid** — all 9 playstyle-grid cells per player are model inputs already.
- **Wicket cascade** — the old rule damped wickets right after one fell, because
  the old engine had no way to know a new batter had arrived. The model sees it
  directly (`striker_balls`, `is_set`, `partnership_balls` all read 0 on his first
  ball). Measured: with the cascade still on, the model *under*-produced all-out
  innings (8.3% vs a real 11.3%) and even three-wicket overs became too rare
  (0.04% vs a real 0.20%). Off, everything landed closer to reality.

## What it looks at (the input) — the exact list

**73 numbers total**, in three groups, plus one more number that's added only
during simulation (not part of training). The single source of truth for all of
this is `ml/etl/schema.py` — if this list and that file ever disagree, the file is
right; it carries a hash that fails loudly if the two drift apart.

### The situation — 41 numbers, genuinely one-per-ball

| feature | plain meaning |
|---|---|
| `over_00` … `over_19` (20 columns) | one column per over, only the current over is "on" — this is how the model learns each over's own shape without assuming a straight-line trend across the innings |
| `ball_in_over` | ball 1–6 of the current over |
| `wickets` | wickets down |
| `balls_remaining` | balls left in the innings |
| `is_second_innings` | chasing or not |
| `rrr` | required run rate (0 if not chasing) |
| `rrr_gt_8` / `rrr_gt_12` / `rrr_gt_16` | is the required rate above 8 / 12 / 16 an over |
| `wickets_x_death` | wickets down, but only counted in the death overs |
| `striker_balls` | how long the batter's been in, capped at 15 (see below for why) |
| `is_set` | has the batter faced 10+ balls |
| `striker_position` | batting order, 1–11 |
| `partnership_balls` | balls faced since the last wicket |
| `bowler_balls` | balls this bowler has sent down this innings |
| `over_in_spell` | how many overs into his current unbroken spell |
| `bat_career_balls` | the batter's career volume — tells the model how much to trust the rest of his stats |
| `bowl_career_balls` | same, for the bowler |
| `nonstriker_ovr` / `nonstriker_sr` | who's at the other end |
| `venue_runs_per_ball` / `venue_wkts_per_ball` | this ground's real scoring/wicket rate |

### The striker — 16 numbers, looked up per player, not per ball

`log_balls, sr, avg, four_rate, six_rate, out_rate, ovr`, plus 9 playstyle-grid
cells (attack / anchor / rotate, × powerplay / middle / death) — the same grid
already used elsewhere in this game.

### The bowler — 16 numbers, same idea, bowling side

`log_balls, eco, avg, sr, wkt_rate, ovr, is_spin`, plus 9 bowling playstyle-grid
cells (attack / contain / defend, × powerplay / middle / death).

Both player profiles come from `data/players_historical.json` and both get
compressed down to just 4 numbers each, with a small learned correction on top to
adjust for anything the raw stats don't capture. This is why a player who's
*never* been seen in training — which the auction pool allows, since anyone can be
drafted — still gets a sensible answer whichever role they're in: they fall back
to what their own career stats alone predict, with the correction at zero.

### The day-factor — 1 number, simulation only, not part of training

A single value drawn once per innings — think "flat road" vs "seaming
minefield" — and held for the whole innings. It's added directly to the odds
during simulation; it never touches the 73 numbers above or the training process.
Without it, every simulated innings drifts toward the same score; real cricket has
much more spread than that, and a fresh random nudge on every single ball can't
produce it — only something that *persists* across an innings can. This is what
closes the gap: innings-score spread of 27.7 without it, 34.0 with it, against a
real 33.8.

A few things are deliberately **excluded** from the input on purpose — not
oversights. See the comment block at the top of `ml/etl/schema.py` for the full
reasoning; the short version is that a few tempting features (like "how many runs
has the batter scored so far") create feedback loops once the model starts
generating its own data during simulation, rather than just reading history.

## Role play — the layer that makes it a game

The model says what the matchup naturally produces. Role play lets the two humans
tilt that, at a cost. It's the only place a player's choice touches a ball, so it's
what separates playing from watching. Code: `ml/runtime/roles.py`.

### The one rule

Every role names buckets that **gain** and buckets that **pay**:

```
T = dial × min(total of gain buckets, total of pay buckets)
  each gain bucket  +=  T × (its share of the gain side)
  each pay  bucket  -=  T × (its share of the pay side)
```

`min()` is a safety guard, not a detail. Consider a tailender using Rotate: his
1s+2s might be 45% while his 4s+6s are only 2.5%. Sizing the trade off the big
side would try to move 6.75% out of a bucket holding 2.5% — a negative
probability. Sizing off the smaller side caps every paying bucket at losing `dial`
of itself, so it can never go negative for anyone.

### The dials, and what they do to each outcome

**30%** for Attack/Defend, **15%** for Rotate/Contain (half strength — a tempo
change, not a gamble).

| batting | 4 | 6 | Out | 0 | 1 | 2 |
|---|---|---|---|---|---|---|
| **Attack** | +30% | +30% | **+30%** | −4.9% | −4.9% | — |
| **Rotate** | −15% | −15% | **unchanged** | — | +2.8% | +2.8% |
| **Defend** | −30% | −30% | **−30%** | +4.9% | +4.9% | — |

| bowling | 4 | 6 | Out | 0 | 1 | 2 |
|---|---|---|---|---|---|---|
| **Attack** | +30% | +30% | **+30%** | −4.9% | −4.9% | — |
| **Contain** | +15% | +15% | **unchanged** | — | −2.8% | −2.8% |
| **Defend** | −30% | −30% | **−30%** | +4.9% | +4.9% | — |

**4s, 6s and Out always move by the exact dial; 0s/1s/2s absorb it and barely
budge.** That asymmetry isn't a bug — it's the same amount of probability moving
either way. The boundary buckets are ~13% of the total and the safe buckets ~78%,
so the same 3.81% is a big slice of one and a rounding error to the other. 3s are
never touched by any role.

### Why 30%

There is no ground truth to fit here — nobody records how hard a batter was
trying, so unlike the day-factor or the calibration constants there's nothing to
recover. These are **chosen**, not discovered.

30% is anchored to a real yardstick so it isn't arbitrary: the league six-rate
moved 4.4% → 7.6% between 2008–15 and 2023–26, i.e. fifteen years of the sport
changing. One Attack call moving sixes by 30% is about a third of that — felt, but
not more powerful than the game reinventing itself.

### Mirrors

Bowling Attack is *numerically identical* to batting Attack, and bowling Defend to
batting Defend. That's correct, not a bug — an attacking bowler and an attacking
batter both make the over more explosive. The opposition shows up in how they
**interact**:

| matchup | result |
|---|---|
| batter Attack vs bowler Defend | cancel **exactly** |
| batter Defend vs bowler Attack | cancel exactly |
| batter Rotate vs bowler Contain | cancel exactly |
| batter Attack vs bowler Attack | compounds — carnage or collapse |
| batter Defend vs bowler Defend | compounds — a dead over |

Cancellation is exact only because both sides' changes are measured off the *same*
gold probs and then added — never applied one after the other, which would let the
first move shift the base the second is measured against.

### Measured effect, 4,000 innings each

| everyone plays… | score | wickets | 4s% | 6s% | out% |
|---|---|---|---|---|---|
| **Defend** | 178.7 | 4.39 | 12.17 | 7.80 | 3.76 |
| **Rotate** | 182.3 | 5.77 | 13.70 | 8.00 | 5.02 |
| **Attack** | 198.0 | 6.68 | 18.69 | 9.81 | 6.11 |
| *real 2023–26* | *181.5* | *6.09* | *13.37* | *7.62* | *5.16* |

Attack vs Defend is a **+19 run / +2.29 wicket** spread — a real decision with a
real cost. Rotate landing almost exactly on real cricket (182.3 vs 181.5) is a good
independent sign the whole thing is grounded: the "play normally" option produces
normal cricket.

**Known imbalance:** Attack is still a slightly favourable bet. It buys +2.6%
boundary chance for +0.6% wicket chance, and all-Attack posts 198 against a real
181 while only reaching 6.68 wickets vs 6.09. Attacking every ball for twenty overs
arguably ought to hurt more. Fixing it means raising Out's dial above 30% on its
own, at the cost of the single-dial simplicity — worth doing if it feels too cheap
in real play.

## What kind of model this actually is

**A straight line.** Not a deep neural network — no hidden layers, no activation
functions. One matrix multiply, then a softmax that turns the result into
percentages summing to 100.

**7,074 numbers**, fitted with plain numpy in about a minute on a CPU. For scale,
that's enormous next to a textbook linear system (a handful of unknowns) and
microscopic next to modern deep learning (billions). It's directly readable: you
could print "over 18 adds this much to the chance of a six."

Where those numbers actually live is the surprising part:

| | | |
|---|---|---|
| `D_bat`, `D_bowl` | 812 × 4 each | **6,496** — a lookup table, one small row per player |
| `B` | 41 × 9 | 369 — the situation |
| `W_bat`/`W_bowl`, `V_bat`/`V_bowl`, `alpha` | small | 209 — connectors |

Over 90% of the model is a per-player lookup table. The situation only touches 369
of the 7,074 numbers.

The classic engine was *already* doing something mathematically equivalent —
multiplying weights is the same as adding in log-space, which is what this does.
The real change isn't the shape of the math, it's that the numbers are fitted from
290,611 real deliveries instead of picked by hand.

A "Wide & Deep" variant (adding a small residual neural net on the GPU to catch
whatever the straight line misses) was designed but **deliberately not built** —
the linear part alone already beats the classic engine comfortably, and cricket's
known predictors carry almost all the signal on their own. With ~350 balls per
player on average, extra flexibility would mostly fit noise.

## How it was trained

- **290,611 real deliveries**, 1,243 IPL matches, 2007–2026 — every match Cricsheet
  has, not a sample.
- **Split randomly by whole match**, not by individual ball, and across *every*
  season rather than holding out only the most recent ones. Splitting by ball would
  let the model see five balls of an over in training and grade itself on the
  sixth — an easy, meaningless win. Holding out only recent seasons would grade the
  model on 2025 cricket while it only ever trained on 2015 cricket, which plays
  very differently (modern IPL scores roughly 1.5 runs per over more than the
  2008–2015 era).
- **Recent matches count more during training** (a 2-season half-life), so the
  model plays like the IPL as it's played *now*, while still using two decades of
  matches to pin down who every player actually is.

## How we know it's good — three separate checks

**1. As a predictor**, graded on held-out matches it never trained on:

| | how wrong it is (lower = better) |
|---|---|
| just guessing the league average | 1.60 |
| **classic engine** | **1.72 — worse than guessing** |
| learned model | 1.53 |

The classic engine is genuinely worse than ignoring the players entirely. Its
biggest miss: on the balls it's most confident are wickets, it says ~17% and
reality is ~10% — nearly 2× overconfident exactly where it matters most.

**2. As a simulator**, running thousands of full innings and comparing the
resulting statistics (average score, how often teams get bowled out, how often
scores pass 200, etc.) against modern real cricket: the model misses **0 of 16**;
the classic engine misses **9 of 16**. Median innings 182 against a real 182,
powerplay run rate exact to two decimals. Run it yourself with
`ml/harness/run_model.py`.

**3. On real chases specifically** — the sharpest test, since a chase that feels
too easy or too hard is the first thing anyone notices. Took 285 real second
innings, replayed each from its real target 40 times, and checked whether the
model's implied win probability matched what actually happened. When the model
says "this chase is 90% likely," the chasing side actually won about 84% of the
time — close, and every probability bucket tracked reality within normal sampling
noise. See `ml/harness/chase_calibration.py`.

## Calibration

Three numbers, fitted against real 2023–26 cricket rather than chosen. Stored in
`ml/artifacts/model_calibration.json`, refitted with
`ml/harness/calibrate_variance.py`.

| | value | what it's matched against |
|---|---|---|
| `calibration` | 1.075 | innings mean |
| `out_calibration` | 1.050 | all-out rate |
| `day_sigma` | 0.178 | innings-score spread |

All three sit close to 1.0, which is the point — the model needs only a light
correction. They were much larger (1.425 / 1.400) before the role rewrite, because
they were quietly compensating for the old role layer's distortions. Once Rotate
and Contain became true exact mirrors, "neutral" started meaning genuinely neutral
and the constants relaxed toward 1.

**These must be refitted if the role dials change.** The 30%/15% dials and these
three constants were fitted together; changing one without the other will drift
the whole thing.

## Open items

- **Attack is slightly underpriced** — see the role play section. The clearest
  candidate for the next change if it feels too cheap in real play.
- **Live-server extras aren't context-aware yet.** The model predicts wide and
  no-ball per ball using the specific bowler, and the offline harness uses that.
  But `src/server.py` decides extras from a flat constant *before* it reaches the
  model, so live play still uses a correct-on-average rate rather than a
  per-ball one. Needs a hook in the server's ball loop.
- **A player's career stats include the very matches used to test the model.**
  A batter's strike-rate input was computed partly from matches in the test set —
  a measured leak (~18% of a typical tested player's career balls), heavily
  diluted by the time it reaches the score, but not precisely quantified. Would
  need career stats rebuilt from training-window matches only.
- **Torch residual head** — designed, deliberately not built. See above.

## A data fix that helped both engines: extras

The classic engine assumed extras happen 4% of the time and are a wide 70% of
that. Real cricket: **3.75%**, and **89%** wides — no-balls are far rarer than the
game assumed, which meant free hits (which only follow a no-ball) were firing
roughly 3× too often. Fixed in `src/engine/simulator.py`. Not a model thing at
all; just a wrong constant, and both engines benefit.

## Where the code lives

| | |
|---|---|
| `ml/etl/` | raw match files → training data, plus `schema.py`, the single source of truth for every model input |
| `ml/runtime/` | what runs during play: the forward pass, role play, venue/player lookups, the engine adapter |
| `ml/train/` | fits the model; `evaluate.py` scores it against the classic engine |
| `ml/harness/` | the simulator behind every check above, plus the calibration fitters |
| `ml/tests/` | 27 tests — feature parity between training and live play, and the role transfer rules |

## Running it

```
python src/server.py                    # the game, on the learned engine
```

Rebuilding from scratch, in order:

```
ml/.venv/Scripts/python -m ml.etl.build_table            # replay → training data
ml/.venv/Scripts/python -m ml.etl.compute_venue_stats    # per-ground scoring rates
ml/.venv/Scripts/python -m ml.train.backbone             # fit the model
ml/.venv/Scripts/python -m ml.harness.calibrate_variance # refit the 3 constants
ml/.venv/Scripts/python -m ml.harness.run_model          # verify vs real cricket
```

The offline steps need `ml/.venv` (see `ml/requirements.txt`). The **server does
not** — it only needs `flask` and `numpy`, reading the trained model from a small
`.npz` file. The classic hand-tuned engine is preserved on the `classic-engine`
branch.
