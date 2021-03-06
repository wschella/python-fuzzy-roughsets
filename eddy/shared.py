# https://minerva.ugent.be/courses2018/C00171902018/document/Project/Fuzzy_LEM2/Rule_induction__2005_.pdf?cidReq=C00171902018
# http://logic.mimuw.edu.pl/Grant2003/prace/FRSMBazan1.pdf
# https://github.com/paudan/scikit-roughsets
# https://minerva.ugent.be/main/document/document.php?curdirpath=%2FProject%2FFuzzy_LEM2&cidReq=C00171902018
# https://www.researchgate.net/publication/238582946_Machine_learning_an_Artificial_Intelligence_approach_Volume_2

from typing import List, Set, Iterable, NewType

import numpy as np

# 0: Temperature
# 1: Headache
# 2: Weakness
# 3: Nausea


class Partition():
    """
    https://en.wikipedia.org/wiki/Partition_of_a_set
    """

    def __init__(self, partition: Iterable[Set[int]], size=None):
        self.partition = partition
        if size:
            self.size = size
        else:
            self.size = len(set().union(*partition))  # type: ignore

    def is_finer(self, partition: 'Partition') -> bool:
        for subset in self.partition:
            for other_subset in partition.partition:
                if subset.issubset(other_subset):
                    break
            else:
                return False
        return True


# def indiscernability_matrix(U, as_indexes=False):
#     """
#     Creates the indiscernability matrix for a given U

#     Parameters
#     ----------
#     U : array_like
#         A 2d integer numpy matrix

#     Returns
#     -------
#     indiscernability_matrix : array_like
#         Given an (n, m) int matrix, return a (n, n, m) bool matrix where an
#         entrie i, j, a is boolean signaling wether U[i][a] differs from U[j][a]
#         in the case where U[i][decision] != U[j][decision]
#     """
#     rows, cols = U.shape
#     M = np.zeros((rows, rows, cols), dtype=bool)
#     for i in range(rows):
#         for j in range(i + 1, rows):

#             # If two rows have the same decision, their entry in the matrix is empty
#             if U[i][cols - 1] == U[j][cols - 1]:
#                 M[i][j] = np.zeros((cols,), dtype=bool)
#             else:
#                 M[i][j] = (U[i] - U[j]).astype(bool)
#                 M[i][j][cols - 1] = False  # We generally do not consider the decision

#     if as_indexes:
#         # Aggregate third axis from a list of booleans into a single python set
#         R = np.apply_along_axis(lambda x: set(np.flatnonzero(x)), 2, M)
#         return (M, R)
#     else:
#         return M


# def indiscernability_relation(U, attributes: List[AttributeIndex]):
#     """
#     Creates the indiscernability relation R of U with respect to attributes

#     Parameters
#     ----------
#     U : array_like
#     attributes : list of int
#         List of attributes to base the indiscernability relation on

#     Returns
#     -------
#     indiscernability_relation : array_like
#         Given an (n, m) int matrix, return an (n, n) boolean matrix where each
#         entry (i, j) represents whether U[i] and U[j] are discernible with
#         with respect to the given attributes
#     """
#     size, _ = U.shape
#     R = np.zeros((size, size), dtype=bool)
#     for i in range(size):
#         for j in range(i + 1, size):
#             R[i][j] = np.any((U[i, attributes] - U[j, attributes]))
#     return R


def elementary_sets(M, attributes: List[int]) -> List[Set[int]]:
    """
    Find the elementary sets (equivalence classes) of M

    Parameters
    ----------
    M : array_like
        A dataset with nominal attributes represented as a (numpy) matrix.
    attributes: list of int
        The attributes to take into account for the indiscernability between two
        rows

    Returns
    -------
    elementary_sets : list of set of int
        Returns a list of the elementary sets of M. This is a list of sets
        containing row indexes with the same equivalence class
    """
    selected_attributes = M[:, attributes]
    unique, inverse = np.unique(selected_attributes, axis=0, return_inverse=True)
    num_equivalence_classes = len(unique)
    equivalence_classes = np.array([set() for _ in range(num_equivalence_classes)], dtype=object)
    for original_index, class_index in enumerate(inverse):
        equivalence_classes[class_index].add(original_index)
    return equivalence_classes.tolist()
