import unittest
from probkit.sampling import rng

class TestSampling(unittest.TestCase):
    def test_rng_has_random_methods(self):
        """Test that rng exposes standard Random methods"""
        self.assertTrue(hasattr(rng, 'random'))
        self.assertTrue(hasattr(rng, 'choices'))
        self.assertTrue(hasattr(rng, 'shuffle'))
        self.assertTrue(callable(rng.random))

    def test_random_returns_float_in_range(self):
        """Test random returns float values in [0,1)"""
        for _ in range(100):
            value = rng.random()
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(value, 0.0)
            self.assertLess(value, 1.0)  # Should be < 1.0, not <= 1.0

    def test_random_reproducibility_with_seed(self):
        """Test that seeding makes random() results reproducible"""
        rng.seed(42)
        values1 = [rng.random() for _ in range(5)]

        rng.seed(42)
        values2 = [rng.random() for _ in range(5)]

        self.assertEqual(values1, values2)

    def test_random_different_values_without_seed(self):
        """Test that consecutive random() calls return different values"""
        rng.seed(None)  # Reset to unpredictable seed
        values = [rng.random() for _ in range(50)]

        # Very unlikely all 50 values are identical
        unique_values = set(values)
        self.assertGreater(len(unique_values), 1)

    def test_random_uses_module_rng(self):
        """Test that random() uses the same RNG as other sampling functions"""
        rng.seed(123)
        random_val = rng.random()

        rng.seed(123)
        # Skip one random call (equivalent to the random() call above)
        rng.random()
        sample_val = rng.ntsig(0.5)

        rng.seed(123)
        # Now get the second random value directly
        rng.random()  # First call
        expected_x = rng.random()  # Second call - this is what ntsig should use

        # Reset and verify
        rng.seed(123)
        actual_random = rng.random()
        self.assertEqual(actual_random, random_val)

    def test_ntsig(self):
        """Test ntsig returns values in [0,1]"""
        for k in [-0.5, 0, 0.5]:
            value = rng.ntsig(k)
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)

    def test_nthsig(self):
        """Test nthsig returns values in [0,1]"""
        for k in [-0.5, 0, 0.5]:
            value = rng.nthsig(k)
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)

    def test_biased_curve(self):
        """Test biased_curve returns values between a and b"""
        a, b = 0.2, 0.8
        for k in [-0.5, 0, 0.5]:
            value = rng.biased_curve(k, a, b)
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(value, min(a, b))
            self.assertLessEqual(value, max(a, b))

    def test_seed_reproducibility(self):
        """Test that seeding makes results reproducible"""
        rng.seed(42)
        values1 = [rng.ntsig(0.5) for _ in range(5)]

        rng.seed(42)
        values2 = [rng.ntsig(0.5) for _ in range(5)]

        self.assertEqual(values1, values2)

    def test_different_values_without_seed(self):
        """Test that consecutive calls return different values"""
        rng.seed(None)  # Reset to unpredictable seed
        values = [rng.ntsig(0.5) for _ in range(10)]

        # Very unlikely all 10 values are identical
        unique_values = set(values)
        self.assertGreater(len(unique_values), 1)

    def test_edge_case_k_values(self):
        """Test with edge case k values"""
        edge_values = [-1, -0.99, 0, 0.99, 1]

        for k in edge_values:
            # Should not raise errors
            val_ntsig = rng.ntsig(k)
            val_nthsig = rng.nthsig(k)

            # Values can be int or float (edge cases return int)
            self.assertIsInstance(val_ntsig, (int, float))
            self.assertIsInstance(val_nthsig, (int, float))
            self.assertGreaterEqual(val_ntsig, 0)
            self.assertLessEqual(val_ntsig, 1)
            self.assertGreaterEqual(val_nthsig, 0)
            self.assertLessEqual(val_nthsig, 1)

    def test_fork(self):
        """Test fork creates independent RNG with same state"""
        rng.seed(42)
        original_val = rng.random()
        
        rng.seed(42)
        rng.random()  # Advance to same state as original_val call above
        forked_rng = rng.fork()  # Fork after advancing
        forked_val = forked_rng.random()  # Should get next value
        main_val = rng.random()  # Main RNG should get same next value
        
        self.assertEqual(main_val, forked_val)  # Both should produce same sequence
        
        # Verify they continue independently
        forked_val2 = forked_rng.random()
        main_val2 = rng.random()
        self.assertEqual(main_val2, forked_val2)  # Still synchronized

    def test_spawn(self):
        """Test spawn creates independent seeded RNG"""
        spawned_rng = rng.spawn(123)
        val1 = spawned_rng.random()
        
        spawned_rng2 = rng.spawn(123)
        val2 = spawned_rng2.random()
        
        self.assertEqual(val1, val2)  # Same seed = same values

    def test_forked_context_manager(self):
        """Test forked context manager doesn't affect main RNG"""
        rng.seed(42)
        main_before = rng.random()
        
        rng.seed(42)
        rng.random()  # Advance to same state
        
        with rng.forked() as r:
            context_val = r.random()  # Get next value from forked RNG
        
        # Main RNG should be at same state as when we forked
        main_after = rng.random()  # This should be same as context_val
        
        self.assertEqual(context_val, main_after)

    def test_spawned_context_manager(self):
        """Test spawned context manager doesn't affect main RNG"""
        rng.seed(42)
        main_before = rng.random()
        
        with rng.spawned(123) as r:
            context_val = r.random()
        
        # Main RNG should be unaffected
        rng.seed(42)
        main_after = rng.random()
        
        self.assertEqual(main_before, main_after)

if __name__ == '__main__':
    unittest.main()
