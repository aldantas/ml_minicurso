#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
import pickle
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
import matplotlib.pyplot as plt


classifiers_dict = {
    'DT': tree.DecisionTreeClassifier,
    'NB': BernoulliNB
}


def apply_experiments(data, classifier_key):
    X_data, y_data = load_svmlight_file(data)
    classifier = classifiers_dict[classifier_key]()
    dump_dict = {}
    for test_size in range(90, 9, -10):
        test_size /= 100
        print(test_size)
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            X_data, y_data, test_size=test_size)
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        score = classifier.score(X_test, y_test)
        cm = confusion_matrix(y_test, y_pred)
        probs = classifier.predict_proba(X_test)
        dump_dict[test_size] = (score, cm, probs)
    with open(classifier_key + '_results.dat', 'wb') as pickle_file:
        pickle.dump(dump_dict, pickle_file)


def main(data):
    for key in classifiers_dict:
        print(key)
        apply_experiments(data, key)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Use: svm.py <data>")

    main(sys.argv[1])
