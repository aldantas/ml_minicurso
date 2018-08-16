
import numpy as np
np.set_printoptions(precision=3, suppress=True)

from sklearn.datasets import load_wine
X_data, y_data = load_wine(return_X_y=True)

from sklearn.svm import SVC
classifier = SVC()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,
                                                    test_size=.3, shuffle=True,
                                                    random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
score = classifier.score(X_test, y_test)
print(score)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
X_scaled_data = scaler.fit_transform(X_data)
print(X_data[0:5,:])
print(X_scaled_data[0:5,:])

X_train, X_test, y_train, y_test = train_test_split(X_scaled_data, y_data,
                                                    test_size=.3, shuffle=True,
                                                    random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
score = classifier.score(X_test, y_test)
print(score)
