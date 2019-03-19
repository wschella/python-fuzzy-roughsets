import unittest

import numpy as np

from eddy.eddy import elementary_sets


class TestElementarySets(unittest.TestCase):
    def test_regular_case(self):
        """
        Test that it can sum a list of integers
        """
        data = np.array([
            [3, 1, 1, 0, 1],
            [2, 1, 0, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 0, 1, 0, 1],
            [2, 0, 0, 0, 0],
            [1, 0, 1, 0, 0]
        ])
        self.assertCountEqual(elementary_sets(data, [0, 4]), [{2, 6}, {3}, {5}, {1, 4}, {0}])
        self.assertCountEqual(elementary_sets(data, list(range(5))),
                              [{0}, {1}, {2}, {3}, {4}, {5}, {6}])


if __name__ == '__main__':
    unittest.main()
