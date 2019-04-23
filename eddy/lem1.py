
from typing import List

from eddy.shared import Partition, elementary_sets


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


def global_covering(U) -> List[int]:
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


def global_covering_part(M, d_part: Partition) -> List[int]:
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
    A: List[int] = list(range(M.shape[1]))
    P: List[int] = list(range(M.shape[1]))
    a_part = Partition(elementary_sets(M, A))
    covering: List[int] = []
    if a_part.is_finer(d_part):
        for a in A:
            Q = P.copy()
            Q.remove(a)
            q_elem = Partition(elementary_sets(M, Q))
            if q_elem.is_finer(d_part):
                P = Q
        covering = P
    return covering
