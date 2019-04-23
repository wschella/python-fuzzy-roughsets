from typing import List

import numpy as np

from eddy.shared import elementary_sets


def get_lower_approximation(U, attributes: List[int], target: List[int]) -> List[int]:
    """ Find the lower approximation of a target set in universe U with respect
    to the indiscernability relation defined by attributes.

    Parameters
    ----------
    U : array-like, shape (n_samples, n_features)
        The universe of cases.
    attributes: array-like, shape (n_attributes,)
        The features/attributes which are the basis of the indiscernability relation
    target: array-like, shape (n_members,)
        The target set we whish to approximate
    Returns
    -------
    lower : ndarray, shape (n_lower_members,)
        A array of indexes of members of the lower approximation
    """
    target_set = set(target)
    elementary = elementary_sets(U, attributes)
    subsets_of_U = filter(lambda s: s.issubset(target_set), elementary)
    return np.array(sorted(list(set.union(*subsets_of_U))), dtype=int)


def get_lower_approximation_mask(U, attributes: List[int], target: List[int]) -> List[bool]:
    """ Find the lower approximation of a target set in universe U with respect
    to the indiscernability relation defined by attributes.

    Parameters
    ----------
    U : array-like, shape (n_samples, n_features)
        The universe of cases.
    attributes: array-like, shape (n_attributes,)
        The features/attributes which are the basis of the indiscernability relation
    target: array-like, shape (n_members,)
        The target set we whish to approximate
    Returns
    -------
    lower : ndarray, shape (n_samples,)
        A mask for U for cases that are in the lower approximation
    """
    lower_approximation = get_lower_approximation(U, attributes, target)
    mask = np.zeros((U.shape[0],), dtype=bool)
    mask[lower_approximation] = True
    return mask
