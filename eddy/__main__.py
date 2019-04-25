
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import scipy

from eddy.lem2 import LEM2Classifier, find_optimal_block

import eddy.datasets as data
import eddy.fuzzyroughsets as frs

names = [
    "LEM2"
]

classifiers = [
    LEM2Classifier(),
]

datasets = [
    # data.paperdata(),
    # data.paperdata2(),
    data.wisconsin(),
]


def test_clean_data(D):
    c = LEM2Classifier()
    c.fit(D[:, :-1], D[:, -1])
    results = c.predict(D[[0, 1, 2], :-1])
    return results


def main():
    for _ds_index, ds in enumerate(datasets):
        ((X, y), ds_name) = ds
        print(f"====================== Evaluating dataset '{ds_name}' ======================")
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=42)
        print("Train")
        print(np.append(X_train, y_train.reshape((y_train.shape[0], 1)), axis=1))

        print("Test")
        print(np.append(X_test, y_test.reshape((y_test.shape[0], 1)), axis=1))

        for c_name, clf in zip(names, classifiers):
            print(f"---------------------- Using {c_name} ----------------------")
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print(classification_report(y_test, y_pred))


def kladblock():
    X = np.array([
        [4, 1, 1, 0],
        [3, 1, 0, 1],
        [2, 0, 0, 0],
        [2, 1, 1, 1],
        [3, 0, 1, 0],
        [3, 0, 0, 0],
        [2, 0, 1, 1]
    ])
    y = np.array([1, 1, 0, 1, 1, 0, 0])
    concept = frs.fuzzy_concept(y, 1)
    print(frs.get_lower_approximation(X, [0, 1, 2, 3], concept))
    # print(X)
    # print(frs.fuzzy_indiscernability(X, [3]))


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    # main()
    kladblock()
