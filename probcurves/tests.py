import unittest
from .probcurves import modified_probability, ntsig, nthsig, biased_curve, transform_range

class Test(unittest.TestCase):
    def test_modified_probability(self):
        # Test base cases
        self.assertEqual(modified_probability(1,ratio=1), 1)
        self.assertEqual(modified_probability(1,ratio=0), 0)
        self.assertEqual(modified_probability(0,ratio=1), 0)
        self.assertEqual(modified_probability(0,ratio=0), 0)
        self.assertEqual(modified_probability(0.5,ratio=float('inf')), 1)
        # Test positional ratio arg
        self.assertLess(modified_probability(0.42,0.9), 0.42)
        self.assertEqual(modified_probability(0.42,1), 0.42)
        self.assertGreater(modified_probability(0.42,1.1), 0.42)
        # Test positional a,b args
        self.assertLess(modified_probability(0.42,2,3), 0.42)
        self.assertEqual(modified_probability(0.42,2,2), 0.42)
        self.assertGreater(modified_probability(0.42,3,2), 0.42)
        # Test division by zero (effective ratio safety)
        self.assertEqual(modified_probability(0.42,1,0), 1)
        self.assertEqual(modified_probability(0.42,1000,0), 1)
        self.assertEqual(modified_probability(0.42,0.0001,0), 1)
        # Test invalid base probability
        with self.assertRaises(ValueError): modified_probability(-0.1,ratio=1)
        with self.assertRaises(ValueError): modified_probability(1.1,ratio=1)
        # Test negative ratio
        with self.assertRaises(ValueError): modified_probability(0.42,ratio=-0.1)
        with self.assertRaises(ValueError): modified_probability(0.42,-0.1)
        with self.assertRaises(ValueError): modified_probability(0.42,-1,0)
        # Test mixing ratio and a/b
        with self.assertRaises(ValueError): modified_probability(0.42,a=1,b=2,ratio=1)
        with self.assertRaises(ValueError): modified_probability(0.42,1,ratio=1)

    def test_ntsig(self):
        # Test edge cases for k
        self.assertEqual(ntsig(-1, 0), 0, "ntsig at k=-1, x=0 should be 0")
        self.assertEqual(ntsig(-1, 0.5), 0.5, "ntsig at k=-1, x=0.5 should be 0.5")
        self.assertEqual(ntsig(-1, 1), 1, "ntsig at k=-1, x=1 should be 1")
        self.assertEqual(ntsig(0, 0), 0, "ntsig at k=0, x=0 should be linear")
        self.assertEqual(ntsig(0, 0.5), 0.5, "ntsig at k=0, x=0.5 should be linear")
        self.assertEqual(ntsig(0, 1), 1, "ntsig at k=0, x=1 should be linear")
        self.assertEqual(ntsig(1, 0), 0, "ntsig at k=1, x=0 should sharply approach 0 or 1")
        self.assertEqual(ntsig(1, 0.5), 0.5, "ntsig at k=1, x=0.5 should be 0.5")
        self.assertEqual(ntsig(1, 1), 1, "ntsig at k=1, x=1 should sharply approach 0 or 1")
        # Test behavior for a sample of x values with specific k values
        for x in [0.1, 0.25, 0.75, 0.9]:
            self.assertTrue(0 <= ntsig(-0.5, x) <= 1, f"ntsig at k=-0.5, x={x} should be within [0,1]")
            self.assertTrue(0 <= ntsig(0.5, x) <= 1, f"ntsig at k=0.5, x={x} should be within [0,1]")
        # Test invalid inputs
        with self.assertRaises(ValueError): ntsig(-1.1, 0.5)
        with self.assertRaises(ValueError): ntsig(1.1, 0.5)
        with self.assertRaises(ValueError): ntsig(0, -0.1)
        with self.assertRaises(ValueError): ntsig(0, 1.1)

    def test_nthsig(self):
        # Test behavior at key points for various k values
        self.assertEqual(nthsig(-1, 0), 0, "nthsig at k=-1, x=0 should be 0")
        self.assertEqual(nthsig(-1, 1), 1, "nthsig at k=-1, x=1 should be 1")
        self.assertEqual(nthsig(0, 0), 0, "nthsig at k=0, x=0 should be linear")
        self.assertEqual(nthsig(0, 1), 1, "nthsig at k=0, x=1 should be linear")
        self.assertEqual(nthsig(1, 0), 0, "nthsig at k=1, x=0 should still be 0")
        self.assertEqual(nthsig(1, 1), 1, "nthsig at k=1, x=1 should still be 1")
        # Test for different k values across the x range
        for x in [0.1, 0.25, 0.75, 0.9]:
            self.assertTrue(0 <= nthsig(-0.5, x) <= 1, f"nthsig at k=-0.5, x={x} should be within [0,1]")
            self.assertTrue(0 <= nthsig(0.5, x) <= 1, f"nthsig at k=0.5, x={x} should be within [0,1]")
        # Test handling of invalid inputs
        with self.assertRaises(ValueError): nthsig(-1.1, 0.5)
        with self.assertRaises(ValueError): nthsig(1.1, 0.5)
        with self.assertRaises(ValueError): nthsig(0, -0.1)
        with self.assertRaises(ValueError): nthsig(0, 1.1)

    def test_biased_curve(self):
        # Test linear behavior
        self.assertEqual(biased_curve(0, 0, 1, 0), 0)
        self.assertEqual(biased_curve(0, 0, 1, 1), 1)
        self.assertEqual(biased_curve(0, 0, 1, 0.5), 0.5)
        # Test bias influence
        self.assertTrue(biased_curve(-1, 0, 1, 0.5) < 0.5, "Bias towards a should lower midpoint value")
        self.assertTrue(biased_curve(1, 0, 1, 0.5) > 0.5, "Bias towards b should raise midpoint value")
        # Test a and b inversion
        self.assertEqual(biased_curve(0, 1, 0, 0), 1)
        self.assertEqual(biased_curve(0, 1, 0, 1), 0)
        self.assertEqual(biased_curve(0, 1, 0, 0.5), 0.5)
        # Test extreme biases
        self.assertTrue(biased_curve(-1, 0, 1, 0.25) < biased_curve(0, 0, 1, 0.25), "Extreme negative bias should significantly lower the value")
        self.assertTrue(biased_curve(1, 0, 1, 0.75) > biased_curve(0, 0, 1, 0.75), "Extreme positive bias should significantly raise the value")
        # Test invalid inputs
        with self.assertRaises(ValueError): biased_curve(0, 0, 1, -0.1)
        with self.assertRaises(ValueError): biased_curve(0, 0, 1, 1.1)

    def test_transform_range(self):
        self.assertEqual(transform_range(0.5, (-1,1), (0,1)), 0.75)
        self.assertEqual(transform_range(0.5, (-1,1), (1,0)), 0.25)
        self.assertEqual(transform_range(0.5, (0,1), (-1,1)), 0)
        self.assertEqual(transform_range(0.5, (0,1), (0,-1)), -0.5)


if __name__ == '__main__':
    unittest.main()
