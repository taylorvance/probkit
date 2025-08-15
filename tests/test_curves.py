import unittest
from probkit.curves import ntsig, nthsig, biased_curve

class TestCurves(unittest.TestCase):
    def test_ntsig(self):
        self.assertEqual(ntsig(0, 0), 0)
        self.assertEqual(ntsig(0, 0.5), 0.5)
        self.assertEqual(ntsig(0, 1), 1)
        self.assertEqual(ntsig(-1, 0.5), 0.5)
        self.assertEqual(ntsig(1, 0.25), 0)
        self.assertEqual(ntsig(1, 0.75), 1)
        with self.assertRaises(ValueError):
            ntsig(2, 0.5)
        with self.assertRaises(ValueError):
            ntsig(0, -0.1)

    def test_nthsig(self):
        self.assertEqual(nthsig(0, 0), 0)
        self.assertEqual(nthsig(0, 1), 1)
        self.assertEqual(nthsig(0, 0.5), 0.5)
        self.assertEqual(nthsig(-1, 0.5), 0)
        self.assertEqual(nthsig(1, 0.5), 1)
        with self.assertRaises(ValueError):
            nthsig(2, 0.5)
        with self.assertRaises(ValueError):
            nthsig(0, -0.1)

    def test_biased_curve(self):
        self.assertEqual(biased_curve(0, 0, 1, 0), 0)
        self.assertEqual(biased_curve(0, 0, 1, 1), 1)
        self.assertEqual(biased_curve(0, 0, 1, 0.5), 0.5)
        self.assertEqual(biased_curve(-1, 0, 1, 0.5), 0)
        self.assertEqual(biased_curve(1, 0, 1, 0.5), 1)

if __name__ == '__main__':
    unittest.main()
