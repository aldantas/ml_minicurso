import numpy as np
np.set_printoptions(precision=3, suppress=True)
from sklearn.model_selection import train_test_split
def clf_runner(X, Y, classifier):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3,
                                                        shuffle=True,
                                                        random_state=42)
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print(score)
