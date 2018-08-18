from sklearn.datasets import load_wine
dataset = load_wine()
print(dataset.DESCR)

X, Y = dataset.data, dataset.target

from clf_runner import clf_runner
from sklearn.svm import SVC
classifier = SVC()
clf_runner(X, Y, classifier)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# print(X[0:5,:])
# print(X_scaled[0:5,:])
clf_runner(X_scaled, Y, classifier)
