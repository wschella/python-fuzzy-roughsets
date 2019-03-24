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

CaseIndex = NewType('CaseIndex', int)
AttributeIndex = NewType('AttributeIndex', int)


class Partition():
    """
    https://en.wikipedia.org/wiki/Partition_of_a_set
    """

    def __init__(self, partition: Iterable[Set[CaseIndex]], size=None):
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


def indiscernability_matrix(U, as_indexes=False):
    """
    Creates the indiscernability matrix for a given U

    Parameters
    ----------
    U : array_like
        A 2d integer numpy matrix

    Returns
    -------
    indiscernability_matrix : array_like
        Given a (n, m) int matrix, return a (n, n, m) bool matrix where an
        entrie i, j, a is boolean signaling wether U[i][a] differs from U[j][a]
        in the case where U[i][decision] != U[j][decision]
    """
    rows, cols = U.shape
    M = np.zeros((rows, rows, cols), dtype=bool)
    for i in range(rows):
        for j in range(i + 1, rows):

            # If two rows have the same decision, their entry in the matrix is empty
            if U[i][cols - 1] == U[j][cols - 1]:
                M[i][j] = np.zeros((cols,), dtype=bool)
            else:
                M[i][j] = (U[i] - U[j]).astype(bool)
                M[i][j][cols - 1] = False  # We generally do not consider the decision

    if as_indexes:
        # Aggregate third axis from a list of booleans into a single python set
        R = np.apply_along_axis(lambda x: set(np.flatnonzero(x)), 2, M)
        return (M, R)
    else:
        return M
