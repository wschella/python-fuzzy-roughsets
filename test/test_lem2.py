import unittest

import numpy as np

import eddy.lem2 as eddy


class TestGetLocalCovering(unittest.TestCase):
    def setUp(self):
        # https://minerva.ugent.be/courses2018/C00171902018/document/Project/Fuzzy_LEM2/Rule_induction__2005_.pdf?cidReq=C00171902018
        self.table1_1 = np.array([
            [3, 1, 1, 0, 1],
            [2, 1, 0, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 0, 1, 0, 1],
            [2, 0, 0, 0, 0],
            [1, 0, 1, 0, 0]
        ])

    def test_paper_case_pos(self):
        self.assertEqual(
            eddy.get_local_covering(self.table1_1[:, :-1], self.table1_1[:, -1] == 1),
            set([frozenset([(1, 1)]), frozenset([(0, 2), (2, 1)])])
        )

    def test_paper_case_neg(self):
        self.assertEqual(
            eddy.get_local_covering(self.table1_1[:, :-1], self.table1_1[:, -1] == 0),
            set([frozenset([(0, 1), (1, 0)]), frozenset([(1, 0), (2, 0)])])
        )


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
            eddy.find_optimal_block(self.table1_1_regular, self.table1_1_regular, set()),
            (3, 0)
        )

    def test_regular_case_proper_subset(self):
        self.assertEqual(
            eddy.find_optimal_block(self.table1_1_regular, self.table1_1_regular[[0, 1]], set()),
            (1, 1)
        )

    def test_always_tie_case(self):
        self.assertEqual(
            eddy.find_optimal_block(self.table1_1_tie, self.table1_1_tie, set()),
            (1, 0)
        )

    def test_always_tie_proper_subset(self):
        self.assertEqual(
            eddy.find_optimal_block(self.table1_1_tie, self.table1_1_tie[[4]], set()),
            (0, 2)
        )

    def test_use_universe(self):
        self.assertEqual(
            eddy.find_optimal_block(self.table1_1_regular, self.table1_1_regular[[0, 1, 2]], set()),
            (1, 1)
        )


if __name__ == '__main__':
    unittest.main()
