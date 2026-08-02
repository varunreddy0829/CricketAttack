"""The live game and the training data must compute every feature identically.

    ml/.venv/Scripts/python -m unittest ml.tests.test_train_serve_parity -v

This exists because `over_in_spell` once diverged: the ETL counted consecutive
overs in an unbroken spell, while the live path counted total overs bowled in the
innings. Same name, same type, same plausible range -- and completely different
meaning, so the model silently answered the wrong question during play.

Nothing about the model catches that. Only a test comparing the two paths does.
"""

from __future__ import annotations

import unittest

from ml.runtime import server_ctx


class SpellCounting(unittest.TestCase):
    """Mirrors the spell logic in ml/etl/replay.py: bowlers alternate ends, so
    consecutive overs by one bowler are exactly two apart."""

    def setUp(self):
        server_ctx.reset()

    def spell(self, name, over):
        return server_ctx._over_in_spell(name, over)

    def test_unbroken_spell_increments(self):
        self.assertEqual(self.spell("A", 0), 1)
        self.assertEqual(self.spell("A", 2), 2)
        self.assertEqual(self.spell("A", 4), 3)

    def test_gap_restarts_the_spell(self):
        self.spell("A", 0)
        self.spell("A", 2)
        # comes back after a long break -- a new spell, not over 3 of the old one
        self.assertEqual(self.spell("A", 12), 1)
        self.assertEqual(self.spell("A", 14), 2)

    def test_is_not_total_overs_bowled(self):
        """The regression this file exists for."""
        for over in (0, 2, 4, 6):
            self.spell("A", over)          # 4 overs bowled, spell length 4
        got = self.spell("A", 16)          # 5th over of the innings, 1st of a new spell
        self.assertEqual(got, 1, "over_in_spell must not be total overs bowled")

    def test_repeat_calls_within_one_over_are_stable(self):
        """Called once per BALL, six times per over -- must not inflate."""
        self.spell("A", 0)
        for _ in range(6):
            self.assertEqual(self.spell("A", 0), 1)
        self.assertEqual(self.spell("A", 2), 2)

    def test_two_bowlers_tracked_independently(self):
        self.assertEqual(self.spell("A", 0), 1)
        self.assertEqual(self.spell("B", 1), 1)
        self.assertEqual(self.spell("A", 2), 2)
        self.assertEqual(self.spell("B", 3), 2)
        self.assertEqual(self.spell("A", 4), 3)

    def test_spells_do_not_carry_across_innings(self):
        self.spell("A", 0)
        self.spell("A", 2)
        server_ctx.reset()
        self.assertEqual(self.spell("A", 4), 1)


class VenueLookup(unittest.TestCase):
    """The game's 10 configured stadiums must all resolve to real per-ground
    stats, not silently fall back to the league average.

    This is the same class of bug as the over_in_spell regression: a lookup
    that quietly returns a plausible-looking placeholder instead of the real,
    per-entity value, with nothing to signal that it happened.
    """

    def test_every_configured_stadium_resolves(self):
        import json
        import os

        from ml.runtime.venues import canonical_ground

        cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "config", "ground_configs.json")
        with open(cfg_path, "r", encoding="utf-8") as fh:
            stadiums = json.load(fh)["stadiums"]

        unresolved = [s["name"] for s in stadiums if canonical_ground(s["name"]) is None]
        self.assertEqual(unresolved, [],
                         f"stadiums with no venue-stats match (falls back to the "
                         f"league average instead of this ground's real rate): "
                         f"{unresolved}")

    def test_venue_rates_differ_by_ground(self):
        """A regression test for the bug itself: every ground returning the same
        number means the lookup silently isn't running."""
        from ml.runtime.server_ctx import venue_rates

        rates = {name: venue_rates(name) for name in
                 ("Wankhede Stadium", "MA Chidambaram Stadium", "Narendra Modi Stadium")}
        self.assertGreater(len(set(rates.values())), 1,
                           f"every ground returned the same rate -- venue_rates is "
                           f"not reading real per-ground stats: {rates}")


class FeatureRowParity(unittest.TestCase):
    """Every model input must be produced by ml/runtime/features.py from both
    sides. `server_ctx` supplies raw state; it must never invent a feature."""

    def test_enriched_ctx_covers_every_argument_build_row_needs(self):
        import inspect

        from ml.runtime.features import build_row

        needed = {
            p for p in inspect.signature(build_row).parameters
            if p != "out"
        }
        # what enrich() supplies, plus what the model's predict() defaults itself
        supplied = {
            "over", "ball_in_over", "wickets", "balls_remaining", "innings_no",
            "score", "target", "striker_balls", "striker_position",
            "partnership_balls", "bowler_balls", "over_in_spell",
            "bat_career_balls", "bowl_career_balls", "ns_ovr", "ns_sr",
            "venue_rpb", "venue_wpb",
        }
        self.assertEqual(needed, supplied,
                         "features.build_row and the live ctx have drifted apart")


if __name__ == "__main__":
    unittest.main()
