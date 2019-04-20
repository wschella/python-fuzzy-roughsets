import numpy as np


def paperdata():
    converters = {
        b'very_high': 4,
        b'high': 3,
        b'normal': 2,
        b'low': 1,
        b'very_low': 0,
        b'yes': 1,
        b'no': 0
    }

    def convert(s):
        return converters[s]

    data = np.genfromtxt(open('data/_paperdata.csv', "rb"),
                         dtype=int, delimiter=",",
                         skip_header=1,
                         converters={0: convert, 1: convert, 2: convert, 3: convert, 4: convert})
    return (data[:, :-1], data[:, -1])
