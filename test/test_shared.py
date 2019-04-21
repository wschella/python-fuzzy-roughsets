import unittest

import numpy as np

import eddy.shared as eddy


class TestElementarySets(unittest.TestCase):
    def setUp(self):
        # https://minerva.ugent.be/courses2018/C00171902018/document/Project/Fuzzy_LEM2/Rule_induction__2005_.pdf?cidReq=C00171902018
        self.table1_1_regular = np.array([
            [3, 1, 1, 0, 1],
            [2, 1, 0, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 0, 1, 0, 1],
            [2, 0, 0, 0, 0],
            [1, 0, 1, 0, 0]
        ])

        self.table1_3_inconsistent = np.array([
            [3, 1, 1, 0, 1],
            [2, 1, 0, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 0, 1, 0, 1],
            [2, 0, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 1, 0, 1],
        ])

    def test_regular_case(self):
        self.assertCountEqual(
            eddy.elementary_sets(self.table1_1_regular, [0, 4]),
            [{2, 6}, {3}, {5}, {1, 4}, {0}])
        self.assertCountEqual(
            eddy.elementary_sets(self.table1_1_regular, list(range(5))),
            [{0}, {1}, {2}, {3}, {4}, {5}, {6}])

    def test_inconsistent_case(self):
        self.assertCountEqual(
            eddy.elementary_sets(self.table1_3_inconsistent, list(range(5))),
            [{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}])

        self.assertCountEqual(
            eddy.elementary_sets(self.table1_3_inconsistent, list(range(4))),
            [{0}, {1}, {2}, {3}, {4}, {5}, {6, 7}])


# class TestIndiscernabilityMatrix(unittest.TestCase):
#     # def test_small_matrix(self):
#     #     data = np.array([
#     #         [1, 0, 0],
#     #         [0, 1, 0],
#     #         [0, 0, 1]
#     #     ])

#     #     result = np.asarray(
#     #         [[set(), set(), {0}],
#     #          [set(), set(), {1}],
#     #          [set(), set(), set()]],
#     #         dtype=object)

#     #     _, actual = eddy.indiscernability_matrix(data, as_indexes=True)
#     #     np.testing.assert_array_equal(actual, result)


# class TestIndiscernabilityRelation(unittest.TestCase):
#     def test_small_matrix(self):
#         data = np.array([
#             [1, 0, 0],
#             [0, 1, 0],
#             [0, 0, 1]
#         ])

#         result = np.asarray(
#             [[0, 1, 1],
#              [0, 0, 1],
#              [0, 0, 0]],
#             dtype=object)

#         actual = eddy.indiscernability_relation(data, list(range(3)))
#         np.testing.assert_array_equal(actual, result)
