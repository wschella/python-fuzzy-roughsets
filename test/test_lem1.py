import unittest

import numpy as np

import eddy.lem1 as eddy


class TestElementarySets(unittest.TestCase):
    def test_regular_case(self):
        data = np.array([
            [3, 1, 1, 0, 1],
            [2, 1, 0, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 0, 1, 0, 1],
            [2, 0, 0, 0, 0],
            [1, 0, 1, 0, 0]
        ])
        self.assertCountEqual(
            eddy.elementary_sets(data, [0, 4]),
            [{2, 6}, {3}, {5}, {1, 4}, {0}])
        self.assertCountEqual(
            eddy.elementary_sets(data, list(range(5))),
            [{0}, {1}, {2}, {3}, {4}, {5}, {6}])

    def test_inconsistent_case(self):
        data = np.array([
            [3, 1, 1, 0, 1],
            [2, 1, 0, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 0, 1, 0, 1],
            [2, 0, 1, 0, 0],  # changed from default
            [1, 0, 1, 0, 0]
        ])
        self.assertCountEqual(
            eddy.elementary_sets(data, list(range(5))),
            [{0}, {1}, {2}, {3}, {4}, {5}, {6}])


class TestGlobalCovering(unittest.TestCase):
    def test_regular_case(self):
        data = np.array([
            [3, 1, 1, 0, 1],
            [2, 1, 0, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 0, 1, 0, 1],
            [2, 0, 0, 0, 0],
            [1, 0, 1, 0, 0]
        ])
        self.assertCountEqual(
            eddy.global_covering(data),
            [0, 2, 3])

    def test_inconsistent_case(self):
        data = np.array([
            [3, 1, 1, 0, 1],
            [2, 1, 0, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 0, 1, 0, 1],
            [2, 0, 1, 0, 0],  # changed from default
            [1, 0, 1, 0, 0]
        ])
        self.assertCountEqual(
            eddy.global_covering(data),
            [])


if __name__ == '__main__':
    unittest.main()
