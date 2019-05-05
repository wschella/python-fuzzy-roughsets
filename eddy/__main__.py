
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import scipy

from eddy.lem2 import LEM2Classifier, find_optimal_block
from eddy.fuzzylem import FuzzyLEM2Classifier

import eddy.datasets as data
import eddy.fuzzyroughsets as frs
import eddy.fuzzylem as fl

names = [
    "LEM2"
]

classifiers = [
    # LEM2Classifier(),
    FuzzyLEM2Classifier(alpha=0.02, beta=0.025),
]

datasets = [
    # data.paperdata(),
    # data.paperdata2(),
    # data.wisconsin(),
    # data.wdbc(),
    data.monk()
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
            # print(y_pred)
            print(classification_report(y_test, y_pred))


def kladblock():
    table1_1 = np.array([
        [3, 1, 1, 0, 1],
        [2, 1, 0, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [2, 0, 1, 0, 1],
        [2, 0, 0, 0, 0],
        [1, 0, 1, 0, 0]
    ])

    X = table1_1[:, :-1]
    y = table1_1[:, -1]
    all_attributes = list(range(X.shape[1]))
    concept = frs.fuzzy_concept(y, 1)
    # print("Concept", concept)
    # lower = frs.get_lower_approximation(X, all_attributes, concept)
    # print("Lower", lower)
    lower = np.array([1, 1, 0, 1, 1, 0, 0])
    covering = fl.get_local_covering(X, lower, alpha=0.05, beta=0.2)
    print("Kladblok covering", covering)


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
    # kladblock()
