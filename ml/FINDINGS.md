# Measured findings

Everything here comes from `ml/harness/`, against 2,427 real IPL innings
(290,611 balls, 2008–2026). Reproduce with `python -m ml.harness.run_baseline`
and `python -m ml.harness.run_lookup`.

## 1. The classic engine, measured for the first time

10,000 innings, real XIs and bowling plans, realistic role mix:

| metric | real | classic | |
|---|---|---|---|
| **all-out rate** | **9.4%** | **63.2%** | 6.8× too often |
| innings mean | 162.6 | 148.5 | −14 runs |
| innings SD | 33.8 | 50.6 | too wide |
| under 120 | 9.3% | 29.6% | 3× |
| dot % | 32.5 | 26.4 | |
| Out % per ball | 4.93 | 7.15 | |
| powerplay RR | 8.08 | 10.20 | |

The per-ball wicket rate is only 2.2 points high, and that compounds over 120
balls into a 6.8× error in the all-out rate. This is the case for measuring
distributions rather than per-ball accuracy, in one line.

It also settles a repo contradiction: [simulator.py:28](../src/engine/simulator.py#L28)
records 37.5% all-out, [CLAUDE.md:56](../CLAUDE.md#L56) claims ~14%. Measured: 48%
at neutral roles, 63% at a realistic mix.

## 2. Two sign errors in `PHASE_EFFECTS`

`python -m ml.runtime.lookup` prints the real per-over distribution. Against
[conditions.py:38-42](../src/engine/conditions.py#L38-L42):

| | engine says | reality |
|---|---|---|
| powerplay `Out` | ×1.10 (more wickets) | 3.9% vs 4.93% league — **0.79×, fewer** |
| powerplay `0` | ×0.95 (fewer dots) | 44% vs 32.5% league — **1.35×, more** |

Both signs are inverted. The powerplay is a *low-wicket, high-dot* phase — bowlers
attack less with the field up, and batters who survive cash in. That single error
explains most of the +2.1 rpo powerplay overshoot.

Also: the real wide/no-ball split is **89/11**, not the engine's 70/30, and wides
are U-shaped by over (4.4% → 2.5% → 4.9%) rather than flat.

The `middle: {}` no-op hides the sharpest discontinuity in the innings — at over 6
the field goes out and fours collapse from 16.8% to 8.0% while singles jump from
30% to 46%.

## 3. Error budget: which layer is actually wrong

Same measured per-over base, ablating what runs on top of it
(`run_lookup.py --no-stages --no-roles`). All-out rate, real = 9.35%:

| configuration | all-out | innings mean | innings SD | metrics off |
|---|---|---|---|---|
| measured base only | 21.6% | 157.4 | **24.2** | 4/16 |
| \+ Stage 4/5 roles | 30.6% | 162.9 | 30.2 | 6/16 |
| \+ Stage 1/2/3 player ratios | 50.4% | 155.3 | 43.1 | 8/16 |
| classic engine (global baseline + phase effects) | 63.0% | 148.5 | 50.6 | 11/16 |

**The Stage 1/2/3 player ratios are the single largest error source** — they add
19.8 points of all-out rate on their own and pull the innings mean *down* 7 runs.
They were tuned against the flat global baseline and don't survive being stacked on
a correct one. This is the strongest argument for replacing them with a fitted
model rather than retuning them.

Role bonuses add 9.0 points of all-out and ~3.5% scoring inflation — close to the
~10% the plan predicted, and small enough for the single global calibration
constant to absorb.

## 4. Two corrections to the plan

**Innings variance: the original prediction was right.** The plan predicted an
iid-per-ball model would give SD ≈ 22–26 against a real 33.8. Measured with a
correct base and nothing on top: **24.2**. Mid-range, as predicted. An earlier
reading of the classic engine's SD of 50.6 as "too wide" was misleading — that
excess is innings terminating early from the runaway wicket rate, not genuine
spread. The day-factor latents are needed.

**A flat base can't fix the all-out rate by itself.** Even with a per-ball Out rate
slightly *below* real (4.50% vs 4.93%), the base-only configuration still bowls
teams out 21.6% of the time against a real 9.4%. A league-average wicket rate
dismisses openers and tailenders alike, so dismissals spread evenly across the
order and more innings reach ten down. Real innings concentrate survival in the top
order. Fixing this needs genuine per-player differentiation — which is exactly what
the learned model's anchored player effects are for, and what Stage 1/2/3 attempts
and gets wrong.

## 4b. The game has shifted hard, and it changes the methodology

`ml/etl/season_shift.py`:

| | runs/ball | run rate | dot% | six% |
|---|---|---|---|---|
| 2008–2015 | 1.250 | 7.50 rpo | 34.6 | 4.41 |
| 2023–2026 | 1.501 | **9.01 rpo** | 29.4 | **7.63** |

**+1.5 runs per over and +73% on sixes.** Two consequences, both of which
invalidated an earlier version of this work:

**The split must span all eras.** An initial chronological split (train ≤2023, test
2025–26) fitted a game that no longer exists and then graded it on one it had never
seen. Replaced with a **random split by MATCH across all seasons** — match-level,
never ball-level, because six balls in an over share batter, bowler, over and
day-state, so a ball-level split tests on near-duplicates of the training rows.
Chronological remains available behind `--split chronological`; it answers a
different question (can this extrapolate forward?) and is genuinely harder.

Recency weighting carries the era: at a **2-season half-life** the model natively
produces 7.38% sixes against a modern 7.62% (it produced 5.53% at a 4-season
half-life), while two decades of matches still pin down player effects.

**The calibration target must be modern.** Aiming the simulator at the all-time
average aims it at 2015:

| | all-time | 2023–26 |
|---|---|---|
| innings mean | 162.6 | **181.5** |
| scores over 200 | 13.4% | **33.5%** |
| dot % | 32.5 | 29.4 |

An earlier run reported the model's 29.4% dot rate as a *miss* against the all-time
target of 32.5%. It was not a miss — the model was right about modern cricket and
the yardstick was wrong. `--since 2023` is now the default for the harness and the
calibration.

## 5. The learned model, scored two ways

**As a predictor**, on 183 held-out matches (43,459 balls, random match-level split
spanning 2007–2026, `ml/train/evaluate.py`). Negative log-likelihood, lower is
better:

| | all 9 classes | legal deliveries only |
|---|---|---|
| prior (constant) | 1.6038 | 1.4891 |
| **classic engine** | **1.7210** | **1.6068** |
| learned backbone | 1.5338 | 1.4267 |

By era — the engine is **worst on the modern game**, which is the game being
played:

| era | prior | classic engine | model |
|---|---|---|---|
| 2008–2015 | 1.5794 | 1.6780 | 1.5084 |
| 2016–2022 | 1.5896 | 1.6740 | 1.5203 |
| **2023–2026** | 1.6661 | **1.8622** | **1.5972** |

**The classic engine is worse than predicting the league average on every ball.**
Its player differentiation isn't merely imperfect — as a probability model it is
actively harmful, because it is confidently wrong.

The reliability curves say where:

| P(wicket) bin | engine predicts | model predicts | observed |
|---|---|---|---|
| lowest | 1.72% | 2.08% | 2.35% |
| highest | **17.33%** | 9.49% | **9.86%** |

The engine's top bin predicts 17.3% when reality is 9.9% — nearly 2× over-confident
at exactly the balls it thinks are most dangerous. That is the mechanism behind the
63% all-out rate in §1, arrived at from a completely independent direction.

On P(six) the engine spans 4.5%→9.9% where reality spans 2.6%→15.3%: it barely
discriminates. The model spans 1.6%→17.0% and tracks the diagonal.

**As a simulator**, 5,000 innings, neutral roles, fitted calibration, scored
against **2023–26** targets (`ml/harness/run_model.py --since 2023`):

| metric | real | classic | model |
|---|---|---|---|
| innings mean | 181.5 | 162.2 | **182.2** |
| innings SD | 37.1 | 37.6 | **37.8** |
| all-out rate | 11.3% | 31.3% | **8.3%** |
| p10 / p50 / p90 | 139 / 182 / 226 | 110 / 166 / 207 | **137 / 181 / 230** |
| scores over 200 | 33.5% | 14.8% | **31.4%** |
| six % | 7.62 | 6.57 | **7.69** |
| four % | 13.37 | 13.17 | **13.70** |
| Out % | 5.16 | 6.06 | **4.92** |
| 50+ scores per innings | 1.07 | 0.90 | **1.08** |
| **metrics outside tolerance** | — | **9/16** | **1/16** |

The single remaining miss is dot % at 26.4 against 29.4 (tolerance ±2.5) — a side
effect of the calibration constants, where pushing the wicket rate up to match
all-out squeezes dots down. A per-phase constant instead of one global would fix
it; that is the v2 item the plan already parked.

Note the classic engine's *scores over 200*: 14.8% against a real 33.5%. It cannot
produce the modern game's ceiling at all.

## 6. The day factor is needed, and the plan's estimate was right

At neutral roles with no day factor, the model's per-ball distribution is nearly
exact (dot 32.07 vs 32.50, four 11.73 vs 12.06, six 5.53 vs 5.50) but the
**innings SD is 27.7 against a real 33.8** — squarely in the 22–26 band the plan
predicted for an iid-per-ball model, nudged up slightly by real lineups and venues.

One scalar drawn per innings and held (`day_sigma = 0.225`) closes it: SD 34.0,
200+ scores 12.3% vs 13.4%, sub-120 10.4% vs 9.3%. Phase-detrended lag-1
autocorrelation of per-over run rates lands at −0.01 against a real −0.00, so
there is no sequence structure left worth chasing — the stopping rule the plan set
for ever revisiting a sequence model is satisfied.

An earlier reading that the day factor might be unnecessary came from measuring SD
at a *realistic* role mix, where the harness's random role sampling injects spread
that real players (who choose deliberately) would not. Fitting at neutral is the
correct reference.

## 7. A harness bug worth recording

`plan.lineup` initially contained only batters who actually came to the crease. A
side that lost 6 wickets produced a 7-name lineup, and the simulator's all-out
threshold is `len(lineup) - 1` — so it called 6 wickets all out, while the real-data
side correctly used 10. This inflated the simulated all-out rate roughly threefold
and made the comparison meaningless.

Fixed by padding each lineup with the rest of the named XI from `info.players`.
Model all-out went 30.0% → 8.6% on that change alone. Worth remembering that when a
simulator and its reference disagree wildly, the measurement is a suspect before
the model is.

## 8. Calibration constants are not a substitute

Fitting the two global constants against the lookup-plus-stages configuration
(`python -m ml.harness.calibrate`) wants `out_calibration = 0.39` and
`calibration = 0.70` — Out cut 61%, scoring cut 30%. Corrections that large aren't
absorbing role inflation, they're papering over the stages being wrong. The
constants are the right tool at the ~10% scale the plan anticipated; they are not
the right tool at this scale.
