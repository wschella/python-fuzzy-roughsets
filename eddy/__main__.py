
import numpy as np
from scikit_roughsets.roughsets import RoughSetsReducer

from eddy.eddy import indiscernability, elementary_sets, global_covering


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
    # print(example_data[:, [3, 0]])
    gc = global_covering(example_data)
    print(example_data)
    print(example_data[:, gc])

    # elem_sets = elementary_sets(example_data, [0, 4])
    rsr = RoughSetsReducer()
    reduct = rsr.reduce(example_data[:, :-1], example_data[:, [-1]])
    print(example_data)
    print(example_data[:, np.array(reduct, dtype=int)])
    # print()
    # print(elem_sets)


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
