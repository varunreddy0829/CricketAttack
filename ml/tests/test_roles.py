"""Role play: pin the exact behaviour we designed, so it can't drift silently.

    ml/.venv/Scripts/python -m unittest ml.tests.test_roles -v
"""

from __future__ import annotations

import unittest

from ml.runtime.roles import ATTACK_DIAL, ROTATE_DIAL, apply_roles

# A realistic distribution (Kohli vs Bumrah, over 10, set), per 1000.
GOLD = {"0": 285.90, "1": 497.65, "2": 87.55, "3": 1.93,
        "4": 87.42, "5": 0.0, "6": 20.92, "Out": 18.64}

# A tailender: almost no boundaries. The case that breaks a naive implementation,
# because the "pay" side is tiny relative to the "gain" side.
TAIL = {"0": 480.0, "1": 400.0, "2": 50.0, "3": 2.0,
        "4": 20.0, "5": 0.0, "6": 5.0, "Out": 43.0}


class Conservation(unittest.TestCase):
    def test_total_always_preserved(self):
        for bat in (None, "attack", "rotate", "defend"):
            for bowl in (None, "attack", "contain", "defend"):
                for base in (GOLD, TAIL):
                    got = sum(apply_roles(base, bat, bowl).values())
                    self.assertAlmostEqual(got, sum(base.values()), places=6,
                                           msg=f"bat={bat} bowl={bowl}")

    def test_never_negative_even_for_a_tailender(self):
        for bat in (None, "attack", "rotate", "defend"):
            for bowl in (None, "attack", "contain", "defend"):
                out = apply_roles(TAIL, bat, bowl)
                for k, v in out.items():
                    self.assertGreaterEqual(v, 0.0, f"{k} went negative: {bat}/{bowl}")


class BattingAttack(unittest.TestCase):
    """The headline numbers, exactly as specified."""

    def setUp(self):
        self.out = apply_roles(GOLD, bat_role="attack")

    def test_boundaries_and_out_each_rise_by_the_full_dial(self):
        for k in ("4", "6", "Out"):
            self.assertAlmostEqual(self.out[k] / GOLD[k], 1.0 + ATTACK_DIAL, places=6,
                                   msg=f"{k} should rise exactly {ATTACK_DIAL:.0%}")

    def test_dots_and_singles_pay_the_same_fraction_as_each_other(self):
        f0 = self.out["0"] / GOLD["0"]
        f1 = self.out["1"] / GOLD["1"]
        self.assertAlmostEqual(f0, f1, places=6)
        self.assertLess(f0, 1.0)

    def test_the_paying_side_barely_moves_because_it_is_much_larger(self):
        # ~4.9%, not 30% -- the whole point of sizing off the smaller side
        drop = 1.0 - self.out["0"] / GOLD["0"]
        self.assertLess(drop, 0.10)

    def test_twos_and_threes_untouched(self):
        for k in ("2", "3"):
            self.assertAlmostEqual(self.out[k], GOLD[k], places=6)


class BattingRotate(unittest.TestCase):
    def test_wicket_chance_is_exactly_unchanged(self):
        """Rotate is the risk-neutral option. The classic engine drifted Out ~8%
        here; this must not."""
        out = apply_roles(GOLD, bat_role="rotate")
        self.assertAlmostEqual(out["Out"], GOLD["Out"], places=9)

    def test_boundaries_fall_by_the_half_dial(self):
        out = apply_roles(GOLD, bat_role="rotate")
        for k in ("4", "6"):
            self.assertAlmostEqual(out[k] / GOLD[k], 1.0 - ROTATE_DIAL, places=6)

    def test_singles_and_twos_rise(self):
        out = apply_roles(GOLD, bat_role="rotate")
        for k in ("1", "2"):
            self.assertGreater(out[k], GOLD[k])


class BattingDefend(unittest.TestCase):
    def test_boundaries_and_out_each_fall_by_the_full_dial(self):
        out = apply_roles(GOLD, bat_role="defend")
        for k in ("4", "6", "Out"):
            self.assertAlmostEqual(out[k] / GOLD[k], 1.0 - ATTACK_DIAL, places=6)

    def test_dots_and_singles_rise(self):
        out = apply_roles(GOLD, bat_role="defend")
        for k in ("0", "1"):
            self.assertGreater(out[k], GOLD[k])


class BowlingContain(unittest.TestCase):
    def test_wicket_chance_is_exactly_unchanged(self):
        """Contain mirrors batting Rotate, so it must be risk-neutral too. The
        classic engine drifted Out ~10% here."""
        out = apply_roles(GOLD, bowl_role="contain")
        self.assertAlmostEqual(out["Out"], GOLD["Out"], places=9)

    def test_chokes_singles_and_concedes_boundaries(self):
        out = apply_roles(GOLD, bowl_role="contain")
        for k in ("1", "2"):
            self.assertLess(out[k], GOLD[k])
        for k in ("4", "6"):
            self.assertGreater(out[k], GOLD[k])


class Mirrors(unittest.TestCase):
    """Matched roles must cancel EXACTLY -- the property that makes the matchup a
    real tug of war rather than an approximate one. Only holds because both deltas
    are measured off the same base and added."""

    def assert_neutral(self, bat, bowl):
        out = apply_roles(GOLD, bat_role=bat, bowl_role=bowl)
        for k in GOLD:
            self.assertAlmostEqual(out[k], GOLD[k], places=6,
                                   msg=f"{bat} vs {bowl} should cancel at {k}")

    def test_attack_vs_defend_cancels(self):
        self.assert_neutral("attack", "defend")

    def test_defend_vs_attack_cancels(self):
        self.assert_neutral("defend", "attack")

    def test_rotate_vs_contain_cancels(self):
        self.assert_neutral("rotate", "contain")

    def test_matched_aggression_compounds(self):
        """Both attacking is NOT neutral -- it should be the wildest over."""
        out = apply_roles(GOLD, bat_role="attack", bowl_role="attack")
        self.assertGreater(out["6"] / GOLD["6"], 1.0 + ATTACK_DIAL)
        self.assertGreater(out["Out"] / GOLD["Out"], 1.0 + ATTACK_DIAL)

    def test_matched_caution_compounds(self):
        out = apply_roles(GOLD, bat_role="defend", bowl_role="defend")
        self.assertLess(out["6"] / GOLD["6"], 1.0 - ATTACK_DIAL)
        self.assertLess(out["Out"] / GOLD["Out"], 1.0 - ATTACK_DIAL)


if __name__ == "__main__":
    unittest.main()
