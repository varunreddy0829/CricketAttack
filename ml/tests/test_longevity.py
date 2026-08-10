"""The proven-player layer: what it must do, and what it provably cannot.

Several of these exist because the obvious implementation was tried and was wrong
in a way that looked right. They pin the reasoning, not just the numbers.
"""

from __future__ import annotations

import json
import os
import unittest

from ml.runtime.longevity import (
    BAT_GAIN,
    BAT_PAY,
    FLOOR,
    LONGEVITY_DIAL,
    PENALTY_SCALE,
    apply_longevity,
    build_scores,
    matchup_strength,
)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLD = {"0": 320.0, "1": 370.0, "2": 70.0, "3": 10.0,
        "4": 120.0, "5": 0.0, "6": 55.0, "Out": 55.0}


def _era_records():
    path = os.path.join(REPO, "data", "eras", "2014_2022", "players.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class Transfer(unittest.TestCase):
    def test_total_is_conserved(self):
        for bs in (-1.0, -0.4, 0.0, 0.4, 1.0):
            for ws in (-1.0, 0.0, 1.0):
                out = apply_longevity(GOLD, bs, ws, dial=LONGEVITY_DIAL)
                self.assertAlmostEqual(sum(out.values()), sum(GOLD.values()), places=6)

    def test_nothing_goes_negative_without_a_clamp(self):
        """apply_longevity has NO output floor -- non-negativity is proved from
        the min() sizing. If that proof ever breaks this is where it shows."""
        for dial in (0.3, 0.6, 1.0):
            for bs, ws in ((1.0, -1.0), (-1.0, 1.0), (0.9, -0.8)):
                out = apply_longevity(GOLD, bs, ws, dial=dial)
                self.assertTrue(all(v >= 0.0 for v in out.values()),
                                f"negative weight at dial {dial}, {bs} vs {ws}")

    def test_a_dial_above_one_raises_instead_of_clipping(self):
        with self.assertRaises(ValueError):
            apply_longevity(GOLD, 1.0, -1.0, dial=1.5)

    def test_equal_players_cancel(self):
        """Two proven players, or two unknowns, get the raw model back."""
        for s in (-0.9, 0.0, 0.9):
            self.assertEqual(apply_longevity(GOLD, s, s, dial=0.5), GOLD)

    def test_losing_costs_less_than_winning_pays(self):
        win = apply_longevity(GOLD, 0.9, -0.9, dial=0.5)
        lose = apply_longevity(GOLD, -0.9, 0.9, dial=0.5)
        gained = win["Out"] - GOLD["Out"]        # negative: Out fell
        lost = lose["Out"] - GOLD["Out"]         # positive: Out rose
        self.assertAlmostEqual(lost / -gained, PENALTY_SCALE, places=6)


class GapHandling(unittest.TestCase):
    """A straight line from 0 to the maximum, rescaled and never clamped."""

    def test_the_response_is_linear(self):
        """gap 0 -> nothing, gap +-2 -> the maximum, and proportional between.

        An earlier version squared this. That was an assumption bolted onto the
        design rather than part of it, and it made the curve arbitrary: a matchup
        57% of the way to the maximum received 32% of the effect.
        """
        self.assertAlmostEqual(matchup_strength(1.0, -1.0), 1.0)
        self.assertAlmostEqual(matchup_strength(-1.0, 1.0), -1.0)
        self.assertAlmostEqual(matchup_strength(0.0, 0.0), 0.0)
        self.assertAlmostEqual(matchup_strength(0.5, 0.0), 0.25)
        # proportional: double the gap, double the strength
        self.assertAlmostEqual(matchup_strength(0.8, 0.0),
                               2 * matchup_strength(0.4, 0.0))

    def test_every_gap_stays_distinct(self):
        """A clamp at 1.0 made these three identical. They must not be."""
        a = matchup_strength(0.94, 0.00)     # gap 0.94
        b = matchup_strength(0.94, -0.98)    # gap 1.92
        c = matchup_strength(1.00, -1.00)    # gap 2.00
        self.assertLess(a, b)
        self.assertLess(b, c)

    def test_strength_never_exceeds_one(self):
        """What makes the output floor unnecessary, and what fixes the dial's
        safe range at [0, 1]."""
        for bs in (-1.0, -0.5, 0.0, 0.5, 1.0):
            for ws in (-1.0, -0.5, 0.0, 0.5, 1.0):
                self.assertLessEqual(abs(matchup_strength(bs, ws)), 1.0)

    def test_the_dial_reads_as_the_fraction_of_out_removed(self):
        """The whole layer in one line: Out falls by dial x strength."""
        for dial in (0.2, 0.5, 0.9):
            for bs, ws in ((1.0, -1.0), (0.5, 0.0), (0.94, -0.17)):
                out = apply_longevity(GOLD, bs, ws, dial=dial)
                expected = GOLD["Out"] * (1.0 - dial * matchup_strength(bs, ws))
                self.assertAlmostEqual(out["Out"], expected, places=9)


class Buckets(unittest.TestCase):
    def test_it_buys_survival_not_tempo(self):
        """Out is the only paying bucket, and every scoring bucket gains the same
        PERCENTAGE -- so the shape of the scoring distribution is frozen and
        strike rate barely moves while balls survived climb."""
        self.assertEqual(BAT_PAY, ("Out",))
        self.assertEqual(set(BAT_GAIN), set(GOLD) - {"Out", "5"})

        out = apply_longevity(GOLD, 1.0, -1.0, dial=LONGEVITY_DIAL)
        sr_before = sum(GOLD[k] * int(k) for k in ("1", "2", "3", "4", "6")) / 1000.0
        sr_after = sum(out[k] * int(k) for k in ("1", "2", "3", "4", "6")) / 1000.0
        balls_before, balls_after = 1000.0 / GOLD["Out"], 1000.0 / out["Out"]

        self.assertLess(abs(sr_after / sr_before - 1.0), 0.05,
                        "strike rate must barely move -- this is not a power boost")
        self.assertGreater(balls_after / balls_before, 1.15,
                           "but the innings must actually get longer")

    def test_every_gaining_bucket_moves_by_the_same_percentage(self):
        out = apply_longevity(GOLD, 0.9, -0.9, dial=0.5)
        pcts = [out[k] / GOLD[k] for k in BAT_GAIN if GOLD[k] > 0]
        self.assertAlmostEqual(max(pcts), min(pcts), places=9)


class Scores(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recs = _era_records()
        s = build_scores(cls.recs)
        cls.bat, cls.bowl = s["bat"], s["bowl"]

    def test_the_biggest_careers_score_highest(self):
        """The career axis is plain runs. It used to be runs x avg x SR, which is
        two-thirds RATE -- it re-decided on what the model already knew and put
        Kohli 4th, below de Villiers."""
        top = [n for n, _ in sorted(self.bat.items(), key=lambda x: -x[1])[:6]]
        for name in ("DA Warner", "V Kohli", "S Dhawan", "KL Rahul"):
            self.assertIn(name, top)

    def test_the_best_bowlers_score_highest(self):
        top = [n for n, _ in sorted(self.bowl.items(), key=lambda x: -x[1])[:6]]
        for name in ("YS Chahal", "JJ Bumrah", "Rashid Khan"):
            self.assertIn(name, top)

    def test_a_player_who_never_bowled_takes_the_floor(self):
        """The regression this rule exists for: Buttler has not bowled a ball in
        this era and scored 0.00 -- i.e. an average bowler, and so rated far above
        Raina, who actually bowled 318 of them for 4 wickets."""
        for name in ("JC Buttler", "DA Warner", "S Dhawan"):
            self.assertEqual(self.bowl[name], FLOOR)
        self.assertGreater(self.bowl["SK Raina"], FLOOR,
                           "Raina DID bowl, so he must be scored, not floored")

    def test_a_player_who_never_batted_takes_the_floor(self):
        floored = [n for n, v in self.bat.items() if v == FLOOR]
        self.assertTrue(floored, "nobody was floored -- the rule is not firing")
        for name in floored:
            rec = next(r for r in self.recs if r["name"] == name)
            self.assertEqual(rec["batting"]["balls"], 0)

    def test_the_scale_is_centred_and_bounded(self):
        vals = list(self.bat.values()) + list(self.bowl.values())
        self.assertGreaterEqual(min(vals), -1.0)
        self.assertLessEqual(max(vals), 1.0)

    def test_specialist_bowlers_are_weak_batters(self):
        for name in ("YS Chahal", "JJ Bumrah", "SL Malinga"):
            self.assertLess(self.bat[name], self.bat["V Kohli"])


class KnownLimits(unittest.TestCase):
    """A rate gap is not this layer's to close. Pinned because the temptation is
    obvious and the cost is paid where the batting table does not show it."""

    KOHLI = {"0": 340.0, "1": 400.0, "2": 70.0, "3": 8.0,
             "4": 105.0, "5": 0.0, "6": 37.0, "Out": 40.0}
    ABDV = {"0": 300.0, "1": 350.0, "2": 70.0, "3": 8.0,
            "4": 140.0, "5": 0.0, "6": 100.0, "Out": 32.0}

    @staticmethod
    def _innings(w):
        rpb = sum(w[k] * int(k) for k in ("1", "2", "3", "4", "6")) / 1000.0
        return rpb * (1000.0 / w["Out"])

    def test_the_faster_scorer_still_leads(self):
        """de Villiers struck at 161 over 3500 balls against Kohli's 130. A
        survival layer does not overturn that, and turning the dial up to try
        spends the realism budget for nothing."""
        for dial in (0.3, 0.55, 0.8, 1.0):
            k = self._innings(apply_longevity(self.KOHLI, 0.97, 0.0, dial=dial))
            a = self._innings(apply_longevity(self.ABDV, 0.94, 0.0, dial=dial))
            self.assertLess(k, a)


if __name__ == "__main__":
    unittest.main()
