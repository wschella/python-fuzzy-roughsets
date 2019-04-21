import unittest

import numpy as np

import eddy.lem2 as eddy


class TestFindOptimalBlock(unittest.TestCase):
    def setUp(self):
        # https://minerva.ugent.be/courses2018/C00171902018/document/Project/Fuzzy_LEM2/Rule_induction__2005_.pdf?cidReq=C00171902018
        self.table1_1_regular = np.array([
            [3, 1, 1, 0],
            [2, 1, 0, 1],
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 0, 1, 0],
            [2, 0, 0, 0],
            [1, 0, 1, 0]
        ])

        self.table1_1_tie = np.array([
            [3, 1, 1, 0],
            [2, 1, 0, 1],
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 0, 1, 0],
            [2, 0, 0, 0],
            [1, 0, 1, 1]
        ])

    def test_regular_case(self):
        self.assertEqual(
            eddy.find_optimal_block(self.table1_1_regular, self.table1_1_regular),
            (3, 0)
        )

    def test_regular_case_proper_subset(self):
        self.assertEqual(
            eddy.find_optimal_block(self.table1_1_regular, self.table1_1_regular[[0, 1]]),
            (1, 1)
        )

    def test_always_tie_case(self):
        self.assertEqual(
            eddy.find_optimal_block(self.table1_1_tie, self.table1_1_tie),
            (1, 0)
        )

    def test_always_tie_case_proper_subset(self):
        self.assertEqual(
            eddy.find_optimal_block(self.table1_1_tie, self.table1_1_tie[[0, 1, 2, 3, 4, 6]]),
            (1, 1)
        )

    def test_use_universe(self):
        self.assertEqual(
            eddy.find_optimal_block(self.table1_1_regular, self.table1_1_regular[[0, 1, 2]]),
            (1, 1)
        )


if __name__ == '__main__':
    unittest.main()
