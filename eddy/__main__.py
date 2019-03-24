
import numpy as np
from scikit_roughsets.roughsets import RoughSetsReducer

import eddy.eddy as eddy

example_data = np.array([
    [3, 1, 1, 0, 1],
    [2, 1, 0, 1, 1],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [2, 0, 1, 0, 1],
    [2, 0, 0, 0, 0],
    [1, 0, 1, 0, 0]
])


def main():
    covering = eddy.global_covering(example_data)
    print(covering)


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
