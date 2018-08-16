#!/usr/bin/python

import sys
from sklearn.datasets import load_svmlight_file, load_wine
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import VarianceThreshold, SelectFpr, SelectKBest

data = '../ml_mini_curso/data'
X_data, y_data = load_svmlight_file(data)
# X_data = X_data.toarray()
# X_data, y_data = load_wine(return_X_y=True)

print((X_data.shape[1]))
# classifier = GaussianNB()
# classifier = KNeighborsClassifier()
classifier = MLPClassifier()

# scaler = StandardScaler()
# X_scaled_data = scaler.fit_transform(X_data)
# selector = VarianceThreshold(threshold=(.7 * (1 - .7)))
selector = SelectFpr()
selector = SelectKBest(k=100)
X_data = selector.fit_transform(X_data, y_data)
print((X_data.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,
                                                    test_size=.3, shuffle=True,
                                                    random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
score = classifier.score(X_test, y_test)
print(score)
# cm = confusion_matrix(y_test, y_pred)
# probs = classifier.predict_proba(X_test)
# print(cm)
