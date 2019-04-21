import unittest

import numpy as np

import eddy.lem1 as eddy


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
