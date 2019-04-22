from typing import Set, List, Tuple, Dict
import collections

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

from eddy.roughsets import get_lower_approximation


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
        print(self.X_, self.classes_)

        all_attributes = list(range(self.X_.shape[0]))
        for class_ in self.classes_:
            concept = self.y_ == class_
            print(concept, class_)
            lower = get_lower_approximation(self.X_, all_attributes, concept)
            covering = get_local_covering(self.X_, lower)
            pass

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


def get_local_covering(U, lower_approximation):
    # Cases in the lower approximation
    B: Set[int] = set(lower_approximation)

    # Elements not yet covered
    G: Set[int] = set(lower_approximation)

    # Found minimal complexes
    Cs: Set[Set[int]] = set()

    def block(attr: int, value: int) -> Set[int]:
        return set(np.where(U[:, attr] == value))

    def blocksetblock(blocks: Set[Set[int]]) -> Set[int]:
        return set.intersection(*blocks)

    # Construct covering
    while G:
        blocks: Set[Set[int]] = set()
        visited: Set[AVPair] = set()

        # Construct minimal complex
        while not blocks or not blocksetblock(blocks).issubset(B):
            (attr, value) = find_optimal_block(U, U[G], visited)
            block = set(np.where(U[:, attr] == value))
            blocks.add(block)
            G.intersection_update(block)
            visited.add((attr, value))

        # Make minimal complex actually minimal
        min_complex = blocks.copy()
        for block in blocks:
            if blocksetblock(min_complex - set([block])).issubset(B):
                min_complex = min_complex - set([block])

        Cs.add(set.intersection(*min_complex))
        G = B - set.union(*Cs)

    # Make covering minimal
    covering: Set[Set[int]] = Cs.copy()
    for blocks in Cs:
        covering_to_test: Set[Set[int]] = covering - set([blocks])
        if set.union(*covering_to_test) == B:
            covering = covering - blocks

    return covering


def find_optimal_block(Universe, Subset, visited_pairs: Set[AVPair]) -> AVPair:
    best_pairs: List[Tuple[int, int]] = []

    visited: Dict[int, Set[int]] = collections.defaultdict(set)
    for attr, value in visited_pairs:
        visited[attr].add(value)

    current_max = 0
    for attr, col in enumerate(Subset.T):
        filters = np.isin(col, np.array(visited[attr]), assume_unique=True, invert=True)
        values, counts, *_ = np.unique(col[filters], return_counts=True)
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
