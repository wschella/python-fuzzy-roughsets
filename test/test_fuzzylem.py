import unittest

import numpy as np

import eddy.fuzzylem as eddy
import eddy.fuzzyroughsets as rs


class TestGetBlock(unittest.TestCase):
    def setUp(self):
        # https://minerva.ugent.be/courses2018/C00171902018/document/Project/Fuzzy_LEM2/Rule_induction__2005_.pdf?cidReq=C00171902018
        self.table1_1 = np.array([
            [3, 1, 1, 0],
            [2, 1, 0, 1],
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 0, 1, 0],
            [2, 0, 0, 0],
            [1, 0, 1, 0]
        ])

    def test_table1_1_regular(self):
        np.testing.assert_array_equal(
            eddy.get_block(self.table1_1, 0, 2),
            np.array([0.5, 0, 0.5, 0.5, 0, 0, 0.5])
        )


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

    def test_table1_1_regular(self):
        X = self.table1_1[:, :-1]
        y = self.table1_1[:, -1]
        all_attributes = list(range(X.shape[1]))
        concept = rs.fuzzy_concept(y, 1)
        lower = rs.get_lower_approximation(X, all_attributes, concept)
        covering = eddy.get_local_covering(X, lower)
        self.assertEqual(
            covering,
            set([frozenset([(1, 1)]), frozenset([(0, 2), (2, 1)])])
        )
