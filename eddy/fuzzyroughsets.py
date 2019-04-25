from typing import List

import numpy as np
import scipy
import sklearn

from eddy.shared import elementary_sets

# TODO: Is target set a thing? or target fuzzy set?

FuzzySet = List[float]


def fuzzy_indiscernability(U, attributes: List[int]) -> List[List[float]]:
    """Construct a fuzzy indiscernability relation with respect to the
    given attributes. Metric used is 1 - L2-normalized standard euclidian distance.

    Parameters
    ----------
    U : array-like, shape (n_samples, n_features)
        The universe of cases
    attributes: array-like, shape (n_attributes,)
        The features/attributes which are the basis of the indiscernability relation.

    Returns
    -------
    I : array-like, shape (n_samples, n_samples)
        Symmetric indiscernability matrix.
        Reflects indiscernability between all cases as row/column pair.
    """
    S = U[:, attributes]  # Cases with only selected attributes
    D = scipy.spatial.distance.cdist(S, S, 'seuclidean')
    I = 1 - sklearn.preprocessing.normalize(D, axis=0, norm='max')
    return I


def fuzzy_concept(y, concept: float) -> FuzzySet:
    """ Define the fuzzy membership of all labels y to some label concept
    """
    y_ = np.resize(y, (y.shape[0], 1))
    dists = scipy.spatial.distance.cdist(y_, [[concept]], 'euclidean')
    indiscernability = 1 - sklearn.preprocessing.normalize(dists, axis=0, norm='max')
    return indiscernability.flatten()


def get_lower_approximation(U, attributes: List[int], target: FuzzySet) -> FuzzySet:
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
        Array of membership degree of all samples
    """
    print(U)
    print(attributes)
    print(target)
    (n_samples, _) = U.shape
    I = fuzzy_indiscernability(U, attributes)
    print(I)
    T = np.tile(target, (n_samples, 1))
    print(T)
    implicator = np.maximum(1 - I, T)  # type: ignore
    print(implicator)
    infimum = np.min(implicator, axis=1)
    return infimum
