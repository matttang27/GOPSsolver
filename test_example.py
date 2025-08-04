import unittest
import numpy as np
from example import cmp, findBestStrategy, findBestCounterplay, calculateEV


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
        # Player A wins (card 3 > card 1, gets 2 points, starts with 1 point diff)
        result = calculateEV([3], [1], 1, [], 2)
        self.assertEqual(result, 1)  # Player A should win
        
        # Player A loses (card 1 < card 3, opponent gets 2 points)
        result = calculateEV([1], [3], 1, [], 2)
        self.assertEqual(result, -1)  # Player A should lose
        
        # Tie case
        result = calculateEV([2], [2], 0, [], 2)
        self.assertEqual(result, 0)  # Should be a tie
    
    def test_calculateEV_two_cards(self):
        """Test Nash equilibrium with two cards returns a matrix"""
        
        result = calculateEV([2,3],[2,3],1,[3],2)
        print(result)
        # Should return a 2x2 numpy array
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 2))
        
        # Check specific matrix values
        self.assertEqual(result[0][0], -1)
        self.assertEqual(result[0][1], 1) 
        self.assertEqual(result[1][0], 0) 
        self.assertEqual(result[1][1], -1)
    
    def test_findBestCounterplay(self):
        """Test that findBestCounterplay doesn't crash (it only prints)"""
        payoff_matrix = np.array([
            [-1, 1],
            [0, -1]
        ])
        
        # This function only prints, so we just test it doesn't crash
        try:
            findBestCounterplay(payoff_matrix, [0.5, 0.5])
            findBestCounterplay(payoff_matrix, [0.7, 0.3])
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
        result = calculateEV([3], [1], 5, [], 0)
        self.assertEqual(result, 1)  # Should still win due to initial point difference
        
        result = calculateEV([1], [3], -5, [], 0)
        self.assertEqual(result, -1)  # Should still lose due to initial point difference


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
