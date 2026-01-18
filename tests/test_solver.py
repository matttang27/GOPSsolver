import unittest
import numpy as np
from solver.solver import calculateEV
from solver.utils import cmp, full, compress_cards, guaranteed
from solver.linprog import findBestStrategy, findBestCounterplay

class TestGOPSFunctions(unittest.TestCase):
    
    def test_cmp_function(self):
        """Test the spaceship operator implementation"""
        # Test greater than
        self.assertEqual(cmp(5, 3), 1)
        
        # Test less than
        self.assertEqual(cmp(3, 5), -1)
        
        # Test equal
        self.assertEqual(cmp(5, 5), 0)
        
        # Test with floats
        self.assertEqual(cmp(3.14, 2.71), 1)
        self.assertEqual(cmp(2.71, 3.14), -1)
        self.assertEqual(cmp(3.14, 3.14), 0)

    # Tests probability distributions up to full(6), which already has accurate calculated results.
    def testBaseEV(self):
        """Test that calculateEV returns known correct probability distributions"""
        # Known correct probability distributions for full(i) vs full(i)
        expected_strategies = [
            [1.0],
            [0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.14442317, 0.01535358, 0.0, 0.0, 0.84022325],
            [0.0, 0.17682883, 0.00787058, 0.0, 0.0, 0.81530059],
            [0.01208932, 0.0, 0.18579355, 0.0, 0.0, 0.0, 0.80211714],
            [0.0, 0.0875914, 0.00944292, 0.11884979, 0.05435713, 0.0, 0.0, 0.72975876],
            [0.03077425, 0.00465836, 0.07138744, 0.02803011, 0.1050596, 0.06116305, 0.0, 0.0, 0.69892719]
        ]

        for i in range(1, 8):
            with self.subTest(i=i):
                result = calculateEV(full(i), full(i), 0, full(i), i - 1, "p")
                expected = np.array(expected_strategies[i-1])
                
                # Check that result is a numpy array with correct length
                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(len(result), i)
                
                # Check that probabilities sum to 1
                self.assertAlmostEqual(sum(result), 1.0, places=6)
                
                # Check that all probabilities are non-negative
                for prob in result:
                    self.assertGreaterEqual(prob, 0.0)
                    self.assertLessEqual(prob, 1.0)
                
                # Check against known values (with tolerance for floating point precision)
                np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-8,
                                         err_msg=f"Strategy for full({i}) doesn't match expected values")

    def test_findBestStrategy_2x2_matrix(self):
        """Test finding best strategy for a 2x2 payoff matrix"""
        payoff_matrix = np.array([
            [-1, 1],
            [0, -1]
        ])
        
        p, v = findBestStrategy(payoff_matrix)
        
        # Check that we got valid results
        self.assertIsNotNone(p)
        self.assertIsNotNone(v)
        
        # Check that probabilities sum to 1
        self.assertAlmostEqual(sum(p), 1.0, places=6)
        
        # Check that all probabilities are non-negative
        for prob in p:
            self.assertGreaterEqual(prob, 0)
            self.assertLessEqual(prob, 1)
    
    def test_findBestStrategy_1x1_matrix(self):
        """Test finding best strategy for a 1x1 matrix (trivial case)"""
        payoff_matrix = np.array([[5]])
        
        p, v = findBestStrategy(payoff_matrix)
        
        self.assertIsNotNone(p)
        self.assertIsNotNone(v)
        self.assertAlmostEqual(p[0], 1.0, places=6)
        self.assertAlmostEqual(v, 5.0, places=6)
    
    def test_calculateEV_single_card(self):
        """Test Nash equilibrium with single cards"""
        # Player A wins (card 3 > card 1, gets prize 2, starts with 1 point diff)
        result = calculateEV((3,), (1,), 1, (2,), 0, "v")
        self.assertEqual(result, 1)  # Player A should win
        
        # Player A loses (card 1 < card 3, opponent gets prize 2)
        result = calculateEV((1,), (3,), 1, (2,), 0, "v")
        self.assertEqual(result, -1)  # Player A should lose
        
        # Tie case
        result = calculateEV((2,), (2,), 0, (2,), 0, "v")
        self.assertEqual(result, 0)  # Should be a tie
    
    def test_calculateEV_two_cards_returns_matrix(self):
        """Test that calculateEV with returnType='m' returns a matrix"""
        result = calculateEV((2, 3), (2, 3), 0, (3, 4), 0, "m")
        
        # Should return a 2x2 numpy array
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 2))
    
    def test_calculateEV_two_cards_returns_strategy(self):
        """Test that calculateEV with returnType='p' returns strategy probabilities"""
        result = calculateEV((2, 3), (2, 3), 0, (3, 4), 0, "p")
        
        # Should return a probability array
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 2)  # 2 cards = 2 probabilities
        
        # Probabilities should sum to 1
        self.assertAlmostEqual(sum(result), 1.0, places=6)
        
        # All probabilities should be non-negative
        for prob in result:
            self.assertGreaterEqual(prob, 0)
            self.assertLessEqual(prob, 1)
    
    def test_calculateEV_two_cards_returns_value(self):
        """Test that calculateEV with returnType='v' returns expected value"""
        result = calculateEV((2, 3), (2, 3), 0, (3, 4), 0, "v")
        
        # Should return a single float value
        self.assertIsInstance(result, (int, float, np.floating))
    
    def test_findBestCounterplay(self):
        """Test that findBestCounterplay doesn't crash (it only prints)"""
        payoff_matrix = np.array([
            [-1, 1],
            [0, -1]
        ])
        
        # This function only prints, so we just test it doesn't crash
        try:
            findBestCounterplay(payoff_matrix, np.array([0.5, 0.5]))
            findBestCounterplay(payoff_matrix, np.array([0.7, 0.3]))
        except Exception as e:
            self.fail(f"findBestCounterplay raised an exception: {e}")

    


class TestEdgeCases(unittest.TestCase):
    
    def test_cmp_with_negative_numbers(self):
        """Test cmp function with negative numbers"""
        self.assertEqual(cmp(-5, -3), -1)
        self.assertEqual(cmp(-3, -5), 1)
        self.assertEqual(cmp(-5, -5), 0)
        self.assertEqual(cmp(-1, 1), -1)
        self.assertEqual(cmp(1, -1), 1)
    
    def test_calculateEV_zero_prize(self):
        """Test Nash equilibrium when prize is 0"""
        result = calculateEV((3,), (1,), 5, (0,), 0, "v")
        self.assertEqual(result, 1)  # Should still win due to initial point difference

        result = calculateEV((1,), (3,), -5, (0,), 0, "v")
        self.assertEqual(result, -1)  # Should still lose due to initial point difference


class TestCompressCards(unittest.TestCase):
    
    def test_compress_cards_basic(self):
        """Test basic compression"""
        cardsA = (2, 8, 12)
        cardsB = (3, 9, 15)
        
        compA, compB = compress_cards(cardsA, cardsB)
        
        self.assertEqual(compA, (1, 3, 5))
        self.assertEqual(compB, (2, 4, 6))
    
    def test_compress_cards_preserves_order(self):
        """Test that relative card ordering is preserved"""
        cardsA = (10, 30)
        cardsB = (20,)
        
        compA, compB = compress_cards(cardsA, cardsB)
        
        # 10 < 20 < 30 should become 1 < 2 < 3
        self.assertEqual(compA, (1, 3))
        self.assertEqual(compB, (2,))
        
        # Verify ordering preserved
        self.assertTrue(compA[0] < compB[0] < compA[1])
    
    def test_compress_cards_identical_sets(self):
        """Test compression with identical card sets"""
        cardsA = (5, 10, 15)
        cardsB = (5, 10, 15)
        
        compA, compB = compress_cards(cardsA, cardsB)
        
        self.assertEqual(compA, (1, 2, 3))
        self.assertEqual(compB, (1, 2, 3))
    
    def test_compress_cards_empty(self):
        """Test compression with empty sets"""
        cardsA = ()
        cardsB = ()
        
        compA, compB = compress_cards(cardsA, cardsB)
        
        self.assertEqual(compA, ())
        self.assertEqual(compB, ())


class TestGuaranteed(unittest.TestCase):
    
    def test_guaranteed_player_a_wins(self):
        """Test when player A is guaranteed to win"""
        cardsA = (4, 5)
        cardsB = (1, 2) 
        pointDiff = 1
        prizes = (1, 2)  # Same length as cards
        
        result = guaranteed(cardsA, cardsB, pointDiff, prizes)
        self.assertEqual(result, 1)
    
    def test_guaranteed_player_b_wins(self):
        """Test when player B is guaranteed to win"""
        cardsA = (1, 2)
        cardsB = (4, 5)
        pointDiff = -1  # A is already behind
        prizes = (1, 2)  # Same length as cards
        
        result = guaranteed(cardsA, cardsB, pointDiff, prizes)
        self.assertEqual(result, -1)
    
    def test_guaranteed_player_b_2(self):
        """B only has one higher card, but it is enough"""
        cardsA = (2, 4)
        cardsB = (1, 5)
        pointDiff = 0
        prizes = (1, 2)
        
        result = guaranteed(cardsA, cardsB, pointDiff, prizes)
        self.assertEqual(result, -1)
    
    def identical(self):
        """Test with identical card sets"""
        cardsA = (1, 2, 3)
        cardsB = (1, 2, 3)
        pointDiff = 0
        prizes = (1, 2, 3)  # Same length as cards
        
        # No cards higher than opponent's max, so no guarantee
        result = guaranteed(cardsA, cardsB, pointDiff, prizes)
        self.assertEqual(result, 0)
    
    def test_guaranteed_edge_case_single_cards(self):
        """Test edge case with single cards"""
        cardsA = (5,)
        cardsB = (1,)
        pointDiff = 0
        prizes = (3,)  # Same length as cards
        
        result = guaranteed(cardsA, cardsB, pointDiff, prizes)
        self.assertEqual(result, 1)
    
    def test_guaranteed_single_card_advantage(self):
        """Test with single high card advantage"""
        cardsA = (1, 2, 10)  # One card (10) higher than B's max (3)
        cardsB = (1, 2, 3)
        pointDiff = 0
        prizes = (1, 2, 5)  # If A gets prize 5, that might be enough
        result = guaranteed(cardsA, cardsB, pointDiff, prizes)
        self.assertEqual(result, 1)
    
    def test_guaranteed_large_point_difference(self):
        """Test with large existing point difference"""
        cardsA = (1, 2)
        cardsB = (3, 4)
        pointDiff = 10  # A is way ahead
        prizes = (1, 1)  # Small remaining prizes
        
        # B is not guaranteed to win, A is already ahead
        result = guaranteed(cardsA, cardsB, pointDiff, prizes)
        self.assertEqual(result, 1)
    
    def test_guaranteed_empty_advantages(self):
        """Test when no player has higher cards than opponent's max"""
        cardsA = (1, 3)
        cardsB = (2, 4)  # B's max is 4, A has no cards > 4
        pointDiff = 0    # A's max is 3, B has no cards > 3  
        prizes = (1, 2, 3)
        
        # greaterThan = 0 + 0 = 0, so no guarantee
        result = guaranteed(cardsA, cardsB, pointDiff, prizes)
        self.assertEqual(result, 0)
    
    def test_guaranteed_tie_breaking_point_diff(self):
        """Test when guarantee calculation exactly equals negative pointDiff"""
        cardsA = (3, 4)
        cardsB = (1, 2)
        pointDiff = -2  # A is behind by 2
        prizes = (1, 1)  # Small prizes
        
        # A has 2 cards higher than B's max (2)  
        # guarantee[2] = sum([1,1]) - sum([]) = 2 - 0 = 2
        # 2 + (-2) = 0, which is NOT > 0, so no guarantee
        result = guaranteed(cardsA, cardsB, pointDiff, prizes)
        self.assertEqual(result, 0)
    
    def test_guaranteed_with_zero_prizes(self):
        """Test with some zero-value prizes"""
        cardsA = (3, 4)
        cardsB = (1, 2)
        pointDiff = 1
        prizes = (0, 0)

        result = guaranteed(cardsA, cardsB, pointDiff, prizes)
        self.assertEqual(result, 1)

def test_guaranteed_never_wrong():
    """Test that guaranteed function never gives wrong answers (false positives)"""
    
    print("\nTesting guaranteed function for false positives...")
    
    # Calculate EVs for small games
    for i in range(1, 8):
        calculateEV(full(i), full(i), 0, full(i), i-1, "v")
    
    cache = calculateEV.cache
    false_positives = []
    
    for key, ev_value in cache.items():
        if len(key) >= 6 and key[5] == "v":
            cardsA, cardsB, pointDiff, prizes, prizeIndex, returnType = key
            remaining_prizes = prizes[prizeIndex:]
            
            guaranteed_result = guaranteed(cardsA, cardsB, pointDiff, remaining_prizes)
            
            # Check for false positives
            if guaranteed_result == 1 and abs(ev_value - 1.0) >= 1e-10:
                false_positives.append((key, ev_value, guaranteed_result, "guaranteed says A wins but EV != 1"))
            elif guaranteed_result == -1 and abs(ev_value + 1.0) >= 1e-10:
                false_positives.append((key, ev_value, guaranteed_result, "guaranteed says A loses but EV != -1"))
    
    print(f"False positives found: {len(false_positives)}")
    
    if false_positives:
        print("=== FALSE POSITIVES (guaranteed function errors) ===")
        for i, (key, ev_value, guaranteed_result, error_type) in enumerate(false_positives[:5]):
            cardsA, cardsB, pointDiff, prizes, prizeIndex, _ = key
            remaining_prizes = prizes[prizeIndex:]
            print(f"{i+1}. {error_type}")
            print(f"   CardsA: {cardsA}, CardsB: {cardsB}")
            print(f"   PointDiff: {pointDiff}, Remaining prizes: {remaining_prizes}")
            print(f"   EV: {ev_value}, Guaranteed: {guaranteed_result}")
    
    return len(false_positives) == 0

def analyze_guaranteed_performance():
    """Analyze how well the guaranteed function performs"""
    
    print("\n=== Guaranteed Function Performance Analysis ===")
    
    # Calculate some games to populate cache
    for i in range(1, 8):
        calculateEV(full(i), full(i), 0, full(i), i-1, "v")
    
    cache = calculateEV.cache
    
    # Count different types of outcomes
    total_states = 0
    guaranteed_wins_in_cache = 0
    guaranteed_losses_in_cache = 0
    guaranteed_detected_wins = 0
    guaranteed_detected_losses = 0
    
    for key, ev_value in cache.items():
        if len(key) >= 6 and key[5] == "v":
            total_states += 1
            cardsA, cardsB, pointDiff, prizes, prizeIndex, _ = key
            remaining_prizes = prizes[prizeIndex:]
            
            guaranteed_result = guaranteed(cardsA, cardsB, pointDiff, remaining_prizes)
            
            if abs(ev_value - 1.0) < 1e-10:  # Actual guaranteed win
                guaranteed_wins_in_cache += 1
                if guaranteed_result == 1:
                    guaranteed_detected_wins += 1
            
            elif abs(ev_value + 1.0) < 1e-10:  # Actual guaranteed loss
                guaranteed_losses_in_cache += 1
                if guaranteed_result == -1:
                    guaranteed_detected_losses += 1
    
    print(f"Total states analyzed: {total_states}")
    print(f"Actual guaranteed wins: {guaranteed_wins_in_cache}")
    print(f"Actual guaranteed losses: {guaranteed_losses_in_cache}")
    print(f"Detected wins: {guaranteed_detected_wins}/{guaranteed_wins_in_cache} ({100*guaranteed_detected_wins/max(guaranteed_wins_in_cache,1):.1f}%)")
    print(f"Detected losses: {guaranteed_detected_losses}/{guaranteed_losses_in_cache} ({100*guaranteed_detected_losses/max(guaranteed_losses_in_cache,1):.1f}%)")
    
    total_guaranteed = guaranteed_wins_in_cache + guaranteed_losses_in_cache
    total_detected = guaranteed_detected_wins + guaranteed_detected_losses
    
    if total_guaranteed > 0:
        print(f"Overall detection rate: {total_detected}/{total_guaranteed} ({100*total_detected/total_guaranteed:.1f}%)")

if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)

    # Run the consistency tests
    print("=== Testing Guaranteed Function Consistency ===")
    no_false_positives = test_guaranteed_never_wrong()
    analyze_guaranteed_performance()
    print(no_false_positives and "No false positives found!" or "False positives detected!")
