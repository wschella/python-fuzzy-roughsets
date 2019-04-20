
import numpy as np
from scikit_roughsets.roughsets import RoughSetsReducer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from eddy.lem2 import LEM2Classifier
from eddy.datasets import paperdata

names = [
    "LEM2"
]

classifiers = [
    LEM2Classifier(),
]

datasets = [
    paperdata()
]


def test_clean_data(D):
    c = LEM2Classifier()
    c.fit(D[:, :-1], D[:, -1])
    results = c.predict(D[[0, 1, 2], :-1])
    return results


def main():
    for _ds_index, ds in enumerate(datasets):
        X, y = ds
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=42)
        # print(X_train, X_test)

        for name, clf in zip(names, classifiers):
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print(name)
            print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
