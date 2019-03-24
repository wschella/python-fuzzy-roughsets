
import numpy as np
from scikit_roughsets.roughsets import RoughSetsReducer

import eddy.shared as shared

example_data = np.array([
    [3, 1, 1, 0, 1],
    [2, 1, 0, 1, 1],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [2, 0, 1, 0, 1],
    [2, 0, 0, 0, 0],
    [1, 0, 1, 0, 0]
])

# Relational calculus slide 53
example_data2 = np.array([
    [2, 1, 0, 1, 1],
    [3, 1, 1, 0, 1],
    [2, 0, 0, 0, 0],
    [2, 0, 0, 0, 1],
    [2, 1, 1, 1, 1],
    [1, 1, 0, 0, 0],
    [1, 0, 1, 1, 1],
    [1, 0, 1, 1, 1]
])

example_data3 = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])


def main():
    _m, i = shared.indiscernability_matrix(example_data2, as_indexes=True)
    print(example_data2)
    print(i)


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
