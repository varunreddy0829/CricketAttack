# Eras — why the game is split, and where OVR comes from

## The problem this solves

Cricket in 2026 is not the game it was in 2010. Measured across the same data the
model trains on:

| | runs/ball | run rate | six rate |
|---|---|---|---|
| 2008–2015 | 1.245 | 7.47 rpo | 4.4% |
| 2023–2026 | 1.501 | **9.01 rpo** | **7.6%** |

One model trained across all nineteen years has to average those, and averaging
them serves neither end. But the sharper problem showed up in the **auction**.

Measured team runs added over a replacement-level opener, career-wide model:

| player | old OVR | runs added |
|---|---|---|
| **Abhishek Sharma** | **80** | **+13.9** |
| N Pooran | 81 | +12.8 |
| V Kohli | 99 | +10.7 |
| **CH Gayle** | **93** | **+7.7** |
| HM Amla | 73 | +9.2 |

OVR was barely correlated with value — the four highest-rated players (99, 98,
95, 93) ranked 4th, 5th, 9th and 7th. A 17cr Gayle delivered less than an 80-rated
modern hitter.

**Not because Gayle was overrated.** He was, for years, the only man in the league
hitting sixes at that rate. His 93 is a fair verdict on 2012 cricket. It is not a
fair price against 2026 bowling, where everyone attacks from ball one. Flattening
his rating to fix the economics would erase what he actually did.

So the game splits instead. Gayle is a legend in 2008–2013 because he genuinely
was; Abhishek Sharma is a marquee buy in 2023–2026 because he genuinely is.

## The eras

Cut where the scoring data actually breaks — runs/ball jumps +0.080 at 2013→2014
and +0.070 then +0.095 at 2022→2023. Definitions live in `config/eras.json`.

| era | balls | draftable | character |
|---|---|---|---|
| **2008–2013** The Early Years | 89k | 217 | 7.4 an over, a six is an event |
| **2014–2022** The Power Shift | 125k | 246 | bats get bigger, the rate creeps up |
| **2023–2026** The Modern Game | 66k | 201 | nine an over, everyone attacks |
| **All-Time Legends** | — | 811 | the classic engine, original OVRs |

An 8-team auction needs 175 players, so every era clears it comfortably. The
100-ball batting / 200-ball bowling cutoff is about whether a *rating* is
trustworthy, not about filling the draft.

Overlap is the point: Gayle appears in eras 1–2, Abhishek Sharma in 2–3,
Kohli/Dhoni/Rohit in all three.

**All-Time is deliberately the classic hand-tuned engine**, not a learned one.
It's the nostalgia mode — a player rated for what they achieved, not for what
they'd do against modern bowling.

## What is era-scoped, and why each one has to be

Not just the model. Five things, each of which produces a real bug if shared:

**The player pool.** Each era's stats are that era's only. Gayle's strike rate is
158.8 in 2008–2013 and 135.5 in 2014–2022.

**The playstyle grids.** Re-percentiled against each era's own population. Gayle's
death-overs attack grid reads **94** in 2008–2013 and **68** in 2014–2022 — one
number honestly saying both that he was untouchable and that the game caught up.

**Venue rates.** A ground's character genuinely inverts. Chepauk was the
*highest*-scoring big venue in 2008–2013 (1.311 runs/ball) and among the *lowest*
by 2023–2026 (1.379 against Wankhede's 1.587). Sharing rates flips its identity.

**League baselines.** `LEAGUE_AVG`'s medians come from a player pool, and an
average 2010 regular is not an average 2025 one (`bat_power_base` 148.7 → 160.2 →
170.8). Scoring a 2008 game against all-time medians judges everyone in it
against cricket that hadn't been played yet.

**Calibration.** Three constants per era, fitted against that era's own innings.

## Where OVR comes from now

**It is measured, not computed.** `ml/train/derive_ovr.py` puts each player into a
real lineup, simulates, and records what the team scores against a
replacement-level baseline. Batting OVR is team runs added; bowling OVR is runs
prevented. Both rescale onto the familiar 55–99 band so the auction's tier cuts
keep working.

Two things this has to get right, both learned the hard way:

**It is not batting average.** An earlier attempt scored batters on expected runs
before dismissal — which ranked Amla (SR 119) 4th and Abhishek Sharma (SR 143)
69th. In T20 the scarce resource is *balls*, not wickets: 35 off 20 beats 45 off
40, and any average-shaped metric says the opposite. Simulating a real innings
gets this right because the 120-ball limit is built in.

**Paired comparison.** Every player is measured on the same innings — same seed,
same lineups, same bowling plans. Unpaired, the noise (±3.5 runs) swamps the
signal (a ~10-run spread across the whole pool). Paired, the same comparison is
stable to ~0.5 runs.

**Anchor the band on the median, not the extremes.** Mapping measured runs
linearly from the pool's worst player to its best hands the entire scale to two
outliers. Bowling has a long thin *bad* tail — a few part-timers conceding 12+
runs over baseline — against a tightly packed good end, so stretched across that
range the **median** 2023–2026 bowler landed on OVR 86, putting 69 of 100 bowlers
above the marquee cut. An auction where two thirds of the pool is marquee has no
tiers at all. Anchoring median→68 and p95→88 restores the pyramid:

| era | marquee | mid | base |
|---|---|---|---|
| 2008–2013 | 62 → **31** | 85 → 78 | 70 → **108** |
| 2023–2026 | 100 → **39** | 76 → 58 | 25 → **104** |

Still linear in runs between the anchors — a 5-run gap is still a 5-run gap; only
the anchor points moved. Both ends clip, so the bad tail compresses into 55
instead of setting the scale for everyone above it, and only 2–3 players per era
clip at 99, leaving the elite properly separated.

Because the measured runs are what's stored, re-tuning those anchors costs
nothing: `derive_ovr --rescale-only` remaps what's already on disk without
re-simulating.

**Shrink by sample size.** The simulation's own noise is tiny, but it faithfully
reproduces uncertainty carried in from the model: a batter the model only saw
for 141 balls has an extreme fitted rate, and the measurement repeats it. Raw,
that put **BCJ Cutting (141 balls) at OVR 90** and Livingstone (329) at 99 in
2014–2022 — above Buttler and Pollard, on a fraction of the evidence. Pulling
measured value toward the median by `balls / (balls + 300)` is the standard
reliability correction, and the same idea the ETL already applies to averages
and economies upstream.

## What the ratings look like

| era | top batters |
|---|---|
| 2008–2013 | Gayle (**99**, career 93), Pietersen (93, career 73), Miller, Dhoni |
| 2014–2022 | de Villiers (**99**), Russell, Hetmyer, Pollard, Buttler |
| 2023–2026 | Klaasen (**99**, career 88), Suryavanshi, **Abhishek Sharma (94)**, Patidar |

The reordering against the old career formula is real but not arbitrary —
r = 0.74 / 0.34 / 0.44 across the three eras.

## Where reputation and measurement disagree

This is a design consequence, not a bug, and it's worth knowing about before it
surprises you in an auction. Team-runs-added rewards scoring **rate**, because
in a 120-ball innings the scarce resource is balls. A high-average, medium-tempo
accumulator can be a great cricketer and still add ~0 runs over a median player
in the same slot.

| player | era | OVR | strike rate | measured |
|---|---|---|---|---|
| V Kohli | 2014–2022 | 70 | 129.9 (**p51**) | −1.3 runs |
| MS Dhoni | 2014–2022 | 69 | 131.6 (p57) | −1.6 runs |
| F du Plessis | 2014–2022 | 69 | 130.4 (p55) | −1.6 runs |

Kohli's strike rate in that era is *exactly* the pool median, so "roughly
replacement level by runs added" is arithmetic, not an opinion. The same logic
that demotes him is the logic that correctly demoted Amla (SR 119) and promoted
Abhishek Sharma — which is the reordering this whole exercise was for.

`ml/check_ovr.py` prints these every run under **WATCH** rather than asserting on
them, so the divergence stays visible instead of being either hidden or baked in
as a fake requirement.

One consequence of smaller pools: era draft sets run short at 8 teams (~78–87
players pulled vs all-time's 118). Auto-fill covers the gap from the undrafted
remainder. This is pool size, not the rating method — the old min/max scale
pulled 81.

## The circularity, and how it's broken

OVR is derived *from* the model, so the model must never take it as an input.
Otherwise it is self-referential, and worse, it creates train/serve skew: null
while training, a real number once the auction needs it.

Era records pin `anchor_ovr`, and everything feeding the model reads
`features.model_ovr()` rather than `batting_ovr` directly. This costs nothing —
OVR was always a lossy summary of career stats the model already has in full.

The same guard fixed a bug that killed training outright:
`record.get("batting_ovr", 55)` returns `None` when the key **exists and is
null**, which lands in a float array as NaN — and the `or 55` fallback downstream
doesn't catch it, because **NaN is truthy in Python**. Every ball's
`nonstriker_ovr` was NaN and all three era models diverged.

## Rebuilding an era

In order — each step depends on the last:

```
ml/.venv/Scripts/python -m ml.etl.era_players --era 2023_2026
ml/.venv/Scripts/python -m ml.etl.compute_venue_stats --eras
ml/.venv/Scripts/python -m ml.etl.build_table       --era 2023_2026
ml/.venv/Scripts/python -m ml.train.backbone        --era 2023_2026
ml/.venv/Scripts/python -m ml.harness.calibrate_variance --era 2023_2026
ml/.venv/Scripts/python -m ml.train.derive_ovr      --era 2023_2026
```

Note the order: OVRs come **last**, after the model exists, because they're
measured through it. An era pool built but not yet derived ships with null OVRs;
the server fills placeholders and prints a warning naming the command to run,
rather than silently serving a pool where every player costs the same.

Recency weighting is **off** inside an era — the era boundary already is the
recency filter, so re-weighting would tilt each era toward its own last season.

## Verifying

```
ml/.venv/Scripts/python -m ml.harness.run_model --era 2023_2026   # vs its own reality
ml/.venv/Scripts/python -m ml.play_all_eras                       # a full match in each
ml/.venv/Scripts/python -m ml.test_era_flow                       # the lobby vote
```
