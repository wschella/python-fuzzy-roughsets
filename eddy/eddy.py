# https://minerva.ugent.be/courses2018/C00171902018/document/Project/Fuzzy_LEM2/Rule_induction__2005_.pdf?cidReq=C00171902018
# http://logic.mimuw.edu.pl/Grant2003/prace/FRSMBazan1.pdf
# https://github.com/paudan/scikit-roughsets
# https://minerva.ugent.be/main/document/document.php?curdirpath=%2FProject%2FFuzzy_LEM2&cidReq=C00171902018

from itertools import combinations

from typing import List, Set, Iterable, Any

import numpy as np

# 0: Temperature
# 1: Headache
# 2: Weakness
# 3: Nausea


class Partition():
    """
    https://en.wikipedia.org/wiki/Partition_of_a_set
    """

    def __init__(self, partition: Iterable[Set[Any]], size=None):
        self.partition = partition
        if size:
            self.size = size
        else:
            self.size = len(set().union(*partition))

    def __lt__(self, partition: Partition):
        for subset in self.partition:


def global_covering(M):
    d_elem = elementary_sets(M, [M.shape[1] - 1])
    return _global_covering(M[:, :-1], d_elem)


def _global_covering(M, d_elem):
    A = list(range(M.shape[1]))
    P = list(range(M.shape[1]))
    a_elem = elementary_sets(M, A)
    covering = set()
    if len(a_elem) > len(d_elem):
        for a in A:
            print(P)
            Q = P.copy()
            Q.remove(a)
            q_elem = elementary_sets(M, Q)
            if len(q_elem) > len(d_elem):
                P = Q
        covering = P
    return covering


def indiscernability(_M, _attributes: List[int]) -> List[List[int]]:
    pass


def elementary_sets(M, attributes: List[int]):
    selected_attributes = M[:, attributes]
    unique, inverse = np.unique(selected_attributes, axis=0, return_inverse=True)
    num_equivalence_classes = len(unique)
    equivalence_classes = np.array([set() for _ in range(num_equivalence_classes)], dtype=object)
    for original_index, class_index in enumerate(inverse):
        equivalence_classes[class_index].add(original_index)
    return equivalence_classes
