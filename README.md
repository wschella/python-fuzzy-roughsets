# Eddy: Python Fuzzy Rough Sets

This is a rudimentary library that implements some basic algorithms related to (fuzzy) rough sets. Originated as a project for the Soft Computing course at Ghent University 2018-2019.

**Warning:** This code is more then likely wrong in multiple places. Please use this only as a rudimentary accompaniement to existing implementations and theory.

## Contents

Main contents of this package is a LEM2 Scikit-Learn Classifier (see `eddy/lem2.py`), and a (not really working) Fuzzy LEM Scikit-Learn Classifier (see `eddy/fuzzylem.py`).

## Setup

This package uses [Pipenv](https://docs.pipenv.org/en/latest/) for dependency management. So it should be nothing more then:

* installing `pipenv`
* installing dependencies with `pipenv install`
* running code with `pipenv run python -m eddy`
