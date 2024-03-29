from sklearn.datasets import load_iris, load_digits
# dataset = load_digits()
dataset = load_iris()

print(dataset.DESCR)
print(dataset.data[:5])
print(dataset.target)

X = dataset.data
Y = dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3,
                                                    shuffle=True,
                                                    random_state=42)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)

print(clf.predict(X_test))
print(Y_test)

print(clf.score(X_test, Y_test))
