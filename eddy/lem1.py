
from typing import List, Set

import numpy as np

from eddy.shared import AttributeIndex, Partition, CaseIndex


def lem1(U):
    """
    Generate a ruleset for dataset U

    Parameters
    ----------
    U : array_like
        A dataset with nominal attributes represented as a (numpy) matrix.
        The last column is assumed to be the decision.
        The dataset is assumed to be consistent.

    Returns
    -------
    rule_set : list of Rule
    """
    covering = global_covering(U)
    specific_rules = covering[:, covering]
    raise Exception("Not impemented yet.")


def global_covering(U) -> List[AttributeIndex]:
    """
    Generate a global covering for a dataset U

    Parameters
    ----------
    U : array_like
        A dataset with nominal attributes represented as a (numpy) matrix.
        The last column is assumed to be the decision.

    Returns
    -------
    global_covering : list
        A list of column indexes forming a global covering for U.
        This will be the empty list in case the dataset is inconsistent.
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
