from eddy.shared import elementary_sets


def lower_approximation(U, attributes, target):
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
    lower : ndarray, shape (n_members,)
        The members of the lower approximation. Is at most as big as target
    """
    target_set = set(target)
    elementary = elementary_sets(U, attributes)
    subsets_of_U = filter(lambda s: s.issubset(target_set), elementary)
    return sorted(list(set.union(*subsets_of_U)))
