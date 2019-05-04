import unittest

import numpy as np

import eddy.fuzzy as eddy


class TestFuzzyIntersection(unittest.TestCase):
    def test_simple(self):
        a = np.array([0.1, 0.3, 0.5, 0.0])
        b = np.array([0.4, 0.3, 0.3, 0.1])
        np.testing.assert_array_equal(
            eddy.fuzzy_intersection(a, b),
            np.array([0.1, 0.3, 0.3, 0.0])
        )


class TestFuzzyComplement(unittest.TestCase):
    def test_simple(self):
        a = np.array([0.1, 0.3, 0.5, 0.0])
        np.testing.assert_array_equal(
            eddy.fuzzy_complement(a),
            np.array([0.9, 0.7, 0.5, 1.0])
        )
