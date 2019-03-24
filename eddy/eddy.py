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


def partition_matrix(M):
    unique, inverse = np.unique(M, axis=0, return_inverse=True)
    print(M)
    print(unique)
    print(inverse)
    pass


def lem1(M):
    covering = global_covering(M)
    return covering


def global_covering(U):
    """
    Generate a global covering for a dataset U

    Parameters
    ----------
    U : array_like
        A dataset with nominal attributes represented as a (numpy) matrix.
        The last column is assumed to be the decision

    Returns
    -------
    global_covering : list
        A list of column indexes forming a global covering for U
    """
    d_part = elementary_sets(U, [U.shape[1] - 1])
    return global_covering_part(U[:, :-1], Partition(d_part))


def global_covering_part(M, d_part: Partition) -> List[AttributeIndex]:
    """
    Generate a global covering for M

    Parameters
    ----------
    M : array_like
        A dataset with nominal attributes represented as a (numpy) matrix.
        We assume the decision to not (!) present.
    d_part : Partition
        A Partitition of M based on the decision ({d}*)

    Returns
    -------
    global_covering : list
        A list of column indexes forming a global covering for M
    """
    A: List[AttributeIndex] = list(range(M.shape[1]))  # type: ignore
    P: List[AttributeIndex] = list(range(M.shape[1]))  # type: ignore
    a_part = Partition(elementary_sets(M, A))
    covering: List[AttributeIndex] = []
    if a_part.is_finer(d_part):
        for a in A:
            Q = P.copy()
            Q.remove(a)
            q_elem = Partition(elementary_sets(M, Q))
            if q_elem.is_finer(d_part):
                P = Q
        covering = P
    return covering


def elementary_sets(M, attributes: List[AttributeIndex]) -> List[Set[CaseIndex]]:
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
    return equivalence_classes
