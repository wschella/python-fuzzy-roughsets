from typing import Set, List, Tuple, Dict
from functools import reduce
import collections
import operator

import numpy as np
import pysnooper

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

from eddy.roughsets import get_lower_approximation_mask


class LEM2Classifier(BaseEstimator, ClassifierMixin):
    """ A classifier which implements the LEM2 rough set algorithm.
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
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        print(np.append(self.X_, self.y_.reshape((y.shape[0], 1)), axis=1))

        all_attributes = list(range(self.X_.shape[0]))
        for class_ in self.classes_:
            concept = np.flatnonzero(self.y_ == class_)
            lower = get_lower_approximation_mask(self.X_, all_attributes, concept)
            covering = get_local_covering(self.X_, lower)
            print(covering)

        # induce rules

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

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        print(closest)
        return self.y_[closest]


AVPair = Tuple[int, int]


@pysnooper.snoop('./logs/local_covering.log')
def get_local_covering(U, lower_approximation_mask):
    # Cases in the lower approximation
    B: Set[int] = set(np.flatnonzero(lower_approximation_mask))

    # Elements not yet covered
    G: Set[int] = set(np.flatnonzero(lower_approximation_mask))

    # Found minimal complexes
    Cs: Set[Set[AVPair]] = set()

    def block_mask(pair: AVPair) -> List[bool]:
        attr, value = pair
        return (U[:, attr] == value)

    def blockset_mask(blocks: Set[AVPair]) -> List[bool]:
        return reduce(
            operator.and_,
            map(block_mask, blocks),
            np.ones((U.shape[0],), dtype=bool)
        )

    def covering_block(covering: Set[Set[AVPair]]) -> Set[int]:
        return set(np.flatnonzero(reduce(
            operator.or_,
            map(blockset_mask, covering),
            np.zeros((U.shape[0],), dtype=bool)
        )))

    def get_complex_block(complex_):
        return set(np.flatnonzero(blockset_mask(complex_)))

    # Construct covering
    while G:
        complex_: Set[AVPair] = set()
        visited: Set[AVPair] = set()

        # Construct minimal complex
        while not complex_ or not get_complex_block(complex_).issubset(B):
            (attr, value) = find_optimal_block(U, U[list(G)], visited)
            block = set(np.where(U[:, attr] == value)[0])
            G = G.intersection(block)
            complex_.add((attr, value))
            visited.add((attr, value))

        # Make minimal complex actually minimal
        min_complex = complex_.copy()
        for av_pair in complex_:
            complex_block = get_complex_block(min_complex - set([av_pair]))
            if complex_block.issubset(B):
                min_complex.remove(av_pair)

        Cs.add(frozenset(min_complex))
        G = B - covering_block(Cs)

    # Make covering minimal
    covering: Set[Set[AVPair]] = Cs.copy()
    for complex_ in Cs:
        covering_to_test: Set[Set[AVPair]] = covering - set([complex_])
        if covering_block(covering_to_test) == B:
            covering = covering - complex_

    return covering


def find_optimal_block(Universe, Subset, visited_pairs: Set[AVPair]) -> AVPair:
    best_pairs: List[Tuple[int, int]] = []

    visited: Dict[int, Set[int]] = collections.defaultdict(set)
    for attr, value in visited_pairs:
        visited[attr].add(value)

    current_max = 0
    for attr, col in enumerate(Subset.T):
        filters = np.isin(col, np.array(list(visited[attr])), invert=True)
        values, counts, *_ = np.unique(col[filters], return_counts=True)

        if values.size == 0:  # All pairs filtered out
            continue

        max_freq = np.max(counts)
        if max_freq > current_max:
            current_max = max_freq
            best_pairs = [(attr, value) for value in values[np.where(counts == max_freq)]]
        else:
            best_pairs += [(attr, value) for value in values[np.where(counts >= current_max)]]

    # No ties
    if len(best_pairs) == 1:
        return best_pairs[0]

    # Ties
    occurrences_in_universe = np.array([
        np.count_nonzero(Universe[:, col] == value)
        for col, value in best_pairs
    ])
    min_freq = np.min(occurrences_in_universe)
    new_best_pairs = np.array(best_pairs)[np.argwhere(
        occurrences_in_universe == min_freq).flatten()]

    # If there's ties again, we just return the first one, if there's no ties
    # we do the same
    return (new_best_pairs[0][0], new_best_pairs[0][1])
