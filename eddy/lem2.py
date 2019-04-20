
from typing import List, Set

import numpy as np

from eddy.shared import AttributeIndex, Partition, CaseIndex


# https://pdfs.semanticscholar.org/e5f1/d026918cc3c1e74a7a347347bb44cc91c293.pdf
def lem2(U):
    """
    Generate a ruleset for dataset U

    Parameters
    ----------
    U : array_like
        A dataset with nominal attributes represented as a (numpy) matrix.
        The last column is assumed to be the decision.

    Returns
    -------
    rule_set : list of Rule
    """
    raise Exception("Not yet implemented")


def lower_approximation(U, eq_rel):
    pass
