import unittest

import numpy as np

import eddy.shared as eddy


class TestIndiscernability(unittest.TestCase):
    def test_small_matrix(self):
        data = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        result = np.asarray(
            [[set(), set(), {0}],
             [set(), set(), {1}],
             [set(), set(), set()]],
            dtype=object)

        _, actual = eddy.indiscernability_matrix(data, as_indexes=True)
        np.testing.assert_array_equal(actual, result)
