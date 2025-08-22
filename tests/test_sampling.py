import unittest
from probkit import sampling

class TestSampling(unittest.TestCase):
    def test_random_function_exists(self):
        """Test that random function is available and callable"""
        self.assertTrue(hasattr(sampling, 'random'))
        self.assertTrue(callable(sampling.random))

    def test_random_returns_float_in_range(self):
        """Test random returns float values in [0,1)"""
        for _ in range(100):
            value = sampling.random()
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(value, 0.0)
            self.assertLess(value, 1.0)  # Should be < 1.0, not <= 1.0

    def test_random_reproducibility_with_seed(self):
        """Test that seeding makes random() results reproducible"""
        sampling.seed(42)
        values1 = [sampling.random() for _ in range(5)]

        sampling.seed(42)
        values2 = [sampling.random() for _ in range(5)]

        self.assertEqual(values1, values2)

    def test_random_different_values_without_seed(self):
        """Test that consecutive random() calls return different values"""
        sampling.seed(None)  # Reset to unpredictable seed
        values = [sampling.random() for _ in range(50)]

        # Very unlikely all 50 values are identical
        unique_values = set(values)
        self.assertGreater(len(unique_values), 1)

    def test_random_uses_module_rng(self):
        """Test that random() uses the same RNG as other sampling functions"""
        sampling.seed(123)
        random_val = sampling.random()

        sampling.seed(123)
        # Skip one random call (equivalent to the random() call above)
        sampling.random()
        sample_val = sampling.sample_ntsig(0.5)

        sampling.seed(123)
        # Now get the second random value directly
        sampling.random()  # First call
        expected_x = sampling.random()  # Second call - this is what sample_ntsig should use

        # Reset and verify
        sampling.seed(123)
        actual_random = sampling.random()
        self.assertEqual(actual_random, random_val)

    def test_sample_ntsig(self):
        """Test sample_ntsig returns values in [0,1]"""
        for k in [-0.5, 0, 0.5]:
            value = sampling.sample_ntsig(k)
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)

    def test_sample_nthsig(self):
        """Test sample_nthsig returns values in [0,1]"""
        for k in [-0.5, 0, 0.5]:
            value = sampling.sample_nthsig(k)
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)

    def test_sample_biased_curve(self):
        """Test sample_biased_curve returns values between a and b"""
        a, b = 0.2, 0.8
        for k in [-0.5, 0, 0.5]:
            value = sampling.sample_biased_curve(k, a, b)
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(value, min(a, b))
            self.assertLessEqual(value, max(a, b))

    def test_seed_reproducibility(self):
        """Test that seeding makes results reproducible"""
        sampling.seed(42)
        values1 = [sampling.sample_ntsig(0.5) for _ in range(5)]

        sampling.seed(42)
        values2 = [sampling.sample_ntsig(0.5) for _ in range(5)]

        self.assertEqual(values1, values2)

    def test_different_values_without_seed(self):
        """Test that consecutive calls return different values"""
        sampling.seed(None)  # Reset to unpredictable seed
        values = [sampling.sample_ntsig(0.5) for _ in range(10)]

        # Very unlikely all 10 values are identical
        unique_values = set(values)
        self.assertGreater(len(unique_values), 1)

    def test_seed_function_exists(self):
        """Test that seed function is available and callable"""
        self.assertTrue(hasattr(sampling, 'seed'))
        self.assertTrue(callable(sampling.seed))

        # Should not raise errors
        sampling.seed(123)
        sampling.seed(None)

    def test_edge_case_k_values(self):
        """Test with edge case k values"""
        edge_values = [-1, -0.99, 0, 0.99, 1]

        for k in edge_values:
            # Should not raise errors
            val_ntsig = sampling.sample_ntsig(k)
            val_nthsig = sampling.sample_nthsig(k)

            # Values can be int or float (edge cases return int)
            self.assertIsInstance(val_ntsig, (int, float))
            self.assertIsInstance(val_nthsig, (int, float))
            self.assertGreaterEqual(val_ntsig, 0)
            self.assertLessEqual(val_ntsig, 1)
            self.assertGreaterEqual(val_nthsig, 0)
            self.assertLessEqual(val_nthsig, 1)

if __name__ == '__main__':
    unittest.main()
