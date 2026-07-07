# tests/test_simulation.py

import unittest
import random
from src.models.player import Batter, Bowler
from src.engine.stats_calculator import (
    BASELINE_WEIGHTS,
    apply_stage1_ovr,
    apply_stage2_strike_rate_economy,
    apply_stage3_wicket_factor
)
from src.engine.intent_handler import apply_intents
from src.engine.simulator import calculate_single_ball, EXTRAS_PROB

class TestCricketMathEngine(unittest.TestCase):
    def setUp(self):
        # Sample Players
        self.kohli = Batter(
            name="Virat Kohli",
            ovr=90,
            career_runs=6000,
            career_balls=4615,
            fours=500,
            sixes=100,
            dismissals=160
        )
        self.russell = Batter(
            name="Andre Russell",
            ovr=85,
            career_runs=2000,
            career_balls=1111,
            fours=140,
            sixes=180,
            dismissals=77
        )
        self.bumrah = Bowler(name="Jasprit Bumrah", ovr=88, eco=6.5, wkt=28, legal_balls=600)
        self.parttimer = Bowler(name="Terrible Parttimer", ovr=50, eco=10.5, wkt=5, legal_balls=150)

        # Stage-2/Stage-3 use ghost-stat smoothed, longevity-aware VOLUME^A x RATE^B
        # scores; league_avg carries the exponents, priors, and league-median
        # baselines (server computes these live from config/baseline_weights.json).
        self.league_avg = {
            # Stage 3
            "prior_sr": 21.2,
            "prior_avg": 27.5,
            "k_wickets": 5.0,
            "k_dismissals": 4.0,
            "threat_a": 0.5,
            "threat_b": 1.0,
            "patience_a": 1.0 / 3.0,
            "patience_b": 1.0,
            "threat_base": 0.302,
            "patience_base": 315.0,
            "patience_floor": 0.15,
            "wicket_damp": 0.2,
            # Stage 2
            "prior_bat_sr": 135.3,
            "prior_eco": 8.38,
            "k_balls_sr": 100.0,
            "k_balls_eco": 60.0,
            "bat_power_a": 0.15,
            "bat_power_b": 1.2,
            "bowl_power_a": 0.15,
            "bowl_power_b": 0.2,
            "bat_power_base": 1013.1,
            "bowl_power_base": 1.692,
            "bat_power_floor": 0.15,
            "bowl_power_floor": 0.15,
            "str_damp": 0.2,
        }

    def test_stage1_probability_sum(self):
        """Test that Stage 1 OVR output weights sum to exactly 1000.0"""
        # Kohli vs Bumrah
        w1_bumrah = apply_stage1_ovr(BASELINE_WEIGHTS, self.kohli, self.bumrah)
        self.assertAlmostEqual(sum(w1_bumrah.values()), 1000.0, places=4)

        # Kohli vs Parttimer
        w1_part = apply_stage1_ovr(BASELINE_WEIGHTS, self.kohli, self.parttimer)
        self.assertAlmostEqual(sum(w1_part.values()), 1000.0, places=4)

    def test_stage2_probability_sum(self):
        """Test that Stage 2 Strike Rate vs Economy output weights sum to exactly 1000.0"""
        w1 = apply_stage1_ovr(BASELINE_WEIGHTS, self.kohli, self.bumrah)

        w2 = apply_stage2_strike_rate_economy(w1, self.kohli, self.bumrah, self.league_avg)
        self.assertAlmostEqual(sum(w2.values()), 1000.0, places=4)

    def test_stage3_probability_sum(self):
        """Test that Stage 3 Wicket Factor output weights sum to exactly 1000.0"""
        w1 = apply_stage1_ovr(BASELINE_WEIGHTS, self.kohli, self.bumrah)
        w2 = apply_stage2_strike_rate_economy(w1, self.kohli, self.bumrah, self.league_avg)

        w3 = apply_stage3_wicket_factor(w2, self.kohli, self.bumrah, self.league_avg)
        self.assertAlmostEqual(sum(w3.values()), 1000.0, places=4)

    def test_intent_meter_attacking(self):
        """Test that batter intent > 50 increases run and wicket weights and decreases dots"""
        w3 = apply_stage3_wicket_factor(BASELINE_WEIGHTS, self.kohli, self.bumrah, self.league_avg)

        # Attacking: 70 intent
        w_att = apply_intents(w3, striker_intent=70, bowler_intent=50)

        self.assertAlmostEqual(sum(w_att.values()), 1000.0, places=4)

        # Verify: 1, 2, 4, 6, Out should increase, 0 should decrease
        for k in ['1', '2', '4', '6', 'Out']:
            if w3[k] > 0.0:
                self.assertGreater(w_att[k], w3[k])
        self.assertLess(w_att['0'], w3['0'])

    def test_intent_meter_defensive(self):
        """Test that batter intent < 50 decreases run and wicket weights and increases dots"""
        w3 = apply_stage3_wicket_factor(BASELINE_WEIGHTS, self.kohli, self.bumrah, self.league_avg)

        # Defensive: 30 intent
        w_def = apply_intents(w3, striker_intent=30, bowler_intent=50)

        self.assertAlmostEqual(sum(w_def.values()), 1000.0, places=4)

        # Verify: 1, 2, 4, 6, Out should decrease, 0 should increase
        for k in ['1', '2', '4', '6', 'Out']:
            if w3[k] > 0.0:
                self.assertLess(w_def[k], w3[k])
        self.assertGreater(w_def['0'], w3['0'])

    def test_probabilities_non_negative(self):
        """Test that all intermediate and final outputs have non-negative weights"""
        w1 = apply_stage1_ovr(BASELINE_WEIGHTS, self.russell, self.parttimer)
        for k, v in w1.items():
            self.assertGreaterEqual(v, 0.0)

        w2 = apply_stage2_strike_rate_economy(w1, self.russell, self.parttimer, self.league_avg)
        for k, v in w2.items():
            self.assertGreaterEqual(v, 0.0)

        w3 = apply_stage3_wicket_factor(w2, self.russell, self.parttimer, self.league_avg)
        for k, v in w3.items():
            self.assertGreaterEqual(v, 0.0)

        # Apply intent
        w_final = apply_intents(w3, striker_intent=90, bowler_intent=80)
        for k, v in w_final.items():
            self.assertGreaterEqual(v, 0.0)

    def test_run_simulation_10000_balls(self):
        """Run 10,000 legal ball simulations for Kohli vs Bumrah to check distributions"""
        outcomes = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, 'Out': 0}
        extras_count = 0
        total_balls_attempted = 0
        
        # Fixed random seed for stability in tests
        random.seed(42)
        
        # Use default balanced intents
        self.kohli.intent = 50
        self.bumrah.intent = 50
        
        legal_balls_target = 10000
        legal_balls_done = 0
        
        while legal_balls_done < legal_balls_target:
            total_balls_attempted += 1
            if random.random() < EXTRAS_PROB:
                extras_count += 1
                continue
            
            outcome = calculate_single_ball(self.kohli, self.bumrah, self.league_avg)
            outcomes[outcome] += 1
            legal_balls_done += 1
            
        print("\n" + "=" * 50)
        print("          10,000 Legal Balls Simulation Results")
        print("          Matchup: Kohli vs Bumrah (Balanced Intent=50)")
        print("=" * 50)
        print(f"Total deliveries attempted: {total_balls_attempted}")
        print(f"Extras (Wides/No balls): {extras_count} ({(extras_count / total_balls_attempted)*100:.2f}%)")
        print("Legal outcome frequencies:")
        for outcome, count in outcomes.items():
            percentage = (count / legal_balls_target) * 100
            print(f"  Outcome '{outcome}': {count} ({percentage:.2f}%)")
        print("=" * 50)
        
        self.assertEqual(legal_balls_done, 10000)

if __name__ == "__main__":
    unittest.main()
