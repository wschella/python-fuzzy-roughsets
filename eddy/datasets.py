import numpy as np


def _paperdata(path):
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

    data = np.genfromtxt(
        open(path, "rb"),
        dtype=int, delimiter=",",
        skip_header=1,
        converters={0: convert, 1: convert, 2: convert, 3: convert, 4: convert}
    )
    return split(data)


def paperdata():
    return (_paperdata('data/_paperdata.csv'), "paperdata")


def paperdata2():
    return (_paperdata('data/_paperdata2.csv'), "paperdata2")


def wisconsin():
    return (split(
        np.genfromtxt(
            open('data/wisconsin683(0:9:0)2.csv', 'rb'),
            dtype=int, delimiter=",",
            skip_header=1,
        )
    ), "wisconsin")


def split(data):
    return data[:, :-1], data[:, -1]
