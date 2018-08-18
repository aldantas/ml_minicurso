from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

X, Y = load_breast_cancer(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3,
                                                    shuffle=True,
                                                    random_state=42)

scale = MinMaxScaler()
ft_sel = SelectKBest()
clf = MLPClassifier(max_iter=500)

pipe = Pipeline(steps=[('scale', scale), ('ft_sel', ft_sel), ('clf', clf)])
pipe.fit(X_train, Y_train)
print(pipe.score(X_test, Y_test))

param_dict = {'ft_sel__k': [3, 5, 10],
              'clf__hidden_layer_sizes': [(50,),(30,20)],
              'clf__activation': ['logistic', 'relu'],
              'clf__learning_rate_init': [0.001, 0.01]}
grid = GridSearchCV(pipe, param_dict, n_jobs=5)
grid.fit(X_train, Y_train)
print(grid.score(X_test, Y_test))
print(grid.best_params_)
