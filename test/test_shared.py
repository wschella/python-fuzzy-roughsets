import unittest

import numpy as np

import eddy.shared as eddy


class TestIndiscernabilityMatrix(unittest.TestCase):
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


class TestIndiscernabilityRelation(unittest.TestCase):
    def test_small_matrix(self):
        data = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        result = np.asarray(
            [[0, 1, 1],
             [0, 0, 1],
             [0, 0, 0]],
            dtype=object)

        actual = eddy.indiscernability_relation(data, list(range(3)))
        np.testing.assert_array_equal(actual, result)
