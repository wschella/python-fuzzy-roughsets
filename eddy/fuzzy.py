from typing import List

import numpy as np

FuzzySet = List[float]


def fuzzy_intersection(A: FuzzySet, B: FuzzySet) -> FuzzySet:
    """
    Standard intersection of fuzzy sets
    """
    return np.minimum(A, B)


def fuzzy_union(A: FuzzySet, B: FuzzySet) -> FuzzySet:
    return np.maximum(A, B)


def fuzzy_complement(A: FuzzySet) -> FuzzySet:
    """
    Standard complement of a fuzzy set
    """
    return 1 - A  # type: ignore


def normal_implicator(A: FuzzySet, B: FuzzySet) -> FuzzySet:
    return np.maximum(1 - A, B)  # type: ignore
