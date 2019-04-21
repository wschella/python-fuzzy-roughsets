import unittest

import numpy as np

import eddy.roughsets as eddy


class TestLowerApproximationMatrix(unittest.TestCase):
    def setUp(self):
        # equivalence (P1, P2, P3, P4, P5) =
        #   [{O1, O2}, {O3, O7, O10}, {O4}, {O5}, {O6}, {O8}, {O9}]
        self.wiki_example = np.array([
            [1, 2, 0, 1, 1],  # O1
            [1, 2, 0, 1, 1],  # O2
            [2, 0, 0, 1, 0],  # O3
            [0, 0, 1, 2, 1],  # O4
            [2, 1, 0, 2, 1],  # O5
            [0, 0, 1, 2, 2],  # O6
            [2, 0, 0, 1, 0],  # O7
            [0, 1, 2, 2, 1],  # O8
            [2, 1, 0, 2, 2],  # O9
            [2, 0, 0, 1, 0],  # O0
        ])

    def test_wiki_example(self):
        target = [0, 1, 2, 3]
        attributes = [0, 1, 2, 3, 4]

        np.testing.assert_array_equal(
            eddy.get_lower_approximation(self.wiki_example, attributes, target),
            [0, 1, 3]
        )
