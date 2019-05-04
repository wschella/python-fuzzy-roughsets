from typing import Set, List, Tuple, Dict, FrozenSet
from functools import reduce
import collections
import operator

import numpy as np
import pysnooper

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

from eddy.fuzzyroughsets import get_lower_approximation, FuzzySet
from eddy.fuzzy import fuzzy_intersection, fuzzy_complement, normal_implicator, fuzzy_union


class FuzzyLEM2Classifier(BaseEstimator, ClassifierMixin):
    """ A classifier which implements a Fuzzy LEM2 rough set algorithm.
    For more information regarding how to build your own classifier, read more
    in the :ref:`User Guide <user_guide>`.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """

    def __init__(self):
        pass

    @pysnooper.snoop('./logs/fit.log')
    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, dtype=int, multi_output=True)

        # TODO: Regression? --> Later

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)  # pylint: disable=attribute-defined-outside-init

        self.X_ = X  # pylint: disable=attribute-defined-outside-init
        self.y_ = y  # pylint: disable=attribute-defined-outside-init
        self.rules_ = np.zeros((self.classes_.size,),  # pylint: disable=attribute-defined-outside-init
                               dtype=object)
        n_cases, n_attributes = self.X_.shape
        all_attributes = list(range(n_attributes))
        for class_index, class_ in enumerate(self.classes_):
            concept = np.flatnonzero(self.y_ == class_)
            lower = get_lower_approximation(self.X_, all_attributes, concept)
            covering = get_local_covering(self.X_, lower)
            self.rules_[class_index] = covering

        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X, dtype=int)

        (n_cases, _) = X.shape
        (n_classes, _) = self.classes_.shape

        prediction = np.full((n_cases,), n_classes, dtype=int)

        covering_degree = np.zeros((n_cases, n_classes), dtype=float)

        for case_i, case in enumerate(X):
            for class_i, _class in enumerate(self.classes_):
                covering = self.rules_[class_i]
                degree = get_covering_degree(self.X_, covering, case)
                covering_degree[case_i, class_i] = degree

        # TODO: Check if works correctly
        prediction = self.classes_[np.argmax(covering_degree)]

        return prediction


AVPair = Tuple[int, int]
Covering = Set[FrozenSet[AVPair]]


def get_covering_degree(U, covering: Covering, case) -> float:
    # TODO
    return 1


def get_local_covering(U, lower_approximation: FuzzySet, alpha=0.1, beta=0.2) -> Covering:
    B: List[float] = np.copy(lower_approximation)
    G: List[float] = np.copy(lower_approximation)

    covering: Covering = set()

    while not covers_concept(U, covering, B, beta):
        complex_: Set[AVPair] = set()
        visited: Set[AVPair] = set()

        while not complex_ or not depends(U, complex_, B, alpha):
            (attr, value) = find_optimal_block(U, G, visited)
            block = get_block(U, attr, value)
            G = fuzzy_intersection(G, block)
            complex_.add((attr, value))
            visited.add((attr, value))

         # Make minimal complex actually minimal
        min_complex = complex_.copy()
        for av_pair in complex_:
            if depends(U, min_complex - set([av_pair]), B, alpha):
                min_complex.remove(av_pair)

        covering.add(frozenset(min_complex))
        G = fuzzy_intersection(B, fuzzy_complement(get_covered(U, covering)))

    return covering


def get_block(U, attr: int, value: float) -> FuzzySet:
    range_ = np.ptp(U[:, attr])
    return np.abs((U[:, attr] - value) / range_)


def get_covered(U, covering: Covering) -> FuzzySet:
    covered = np.zeros((U.shape[0],), dtype=float)
    for complex_ in covering:
        block = get_complex_block(U, complex_)  # type: ignore
        covered = fuzzy_union(covered, block)
    return covered


def covers_concept(U, covering: Covering, concept: FuzzySet, beta: float) -> bool:
    covered = get_covered(U, covering)
    return np.all(covered - concept <= beta)  # type: ignore


def get_complex_block(U, complex_: Set[AVPair]) -> FuzzySet:
    block = np.ones((U.shape[0],), dtype=float)
    for attr, value in complex_:
        block = fuzzy_intersection(block, get_block(U, attr, value))
    return block


def depends(U, complex_: Set[AVPair], concept: FuzzySet, alpha: float) -> bool:
    block = get_complex_block(U, complex_)
    subset = normal_implicator(block, concept)
    return np.all(subset >= alpha)  # type: ignore


def find_optimal_block(U, Sub: FuzzySet, visited_pairs: Set[AVPair]) -> AVPair:
    # TODO
    raise Exception("Not implemented")
    return (0, 0)
