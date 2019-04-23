
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from eddy.lem2 import LEM2Classifier, find_optimal_block
import eddy.datasets as data
from eddy.datasets import paperdata, paperdata2

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
    X, y = paperdata2()
    # print(X)
    # print(most_frequent(X))
    # print(find_optimal_block(X, X, set()))
    # (unique, count) = np.unique(X, return_counts=True, axis=1)
    # print(unique)
    # print(count)


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
    # kladblock()
