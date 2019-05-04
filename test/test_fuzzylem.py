import unittest

import numpy as np

import eddy.fuzzylem as eddy


class TestGetBlock(unittest.TestCase):
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

    def test_table1_1_regular(self):
        np.testing.assert_array_equal(
            eddy.get_block(self.table1_1_regular, 0, 2),
            np.array([0.5, 0, 0.5, 0.5, 0, 0, 0.5])
        )
