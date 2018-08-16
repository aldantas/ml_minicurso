import pandas
import numpy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import VarianceThreshold, SelectFpr, SelectKBest

seed = 42

# 1. Number of times pregnant
# 2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 3. Diastolic blood pressure (mm Hg)
# 4. Triceps skin fold thickness (mm)
# 5. 2-Hour serum insulin (mu U/ml)
# 6. Body mass index (weight in kg/(height in m)^2)
# 7. Diabetes pedigree function
# 8. Age (years)
# 9. Class variable (0 or 1)
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv('/home/aldantas/Bio/ml/diabetes.dat', names=names)
data = dataframe.values
# separate array into input and output components
X = numpy.array(data[:,0:8])
Y = numpy.array(data[:,8])

scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
# summarize transformed data

selector = SelectFpr()
# selector = SelectKBest(k=100)
print(X.shape)
X = selector.fit_transform(X, Y)
print(X.shape)

numpy.set_printoptions(precision=3, suppress=True)
print(type(X))
print(type(rescaledX))
print(X[0:5,:])
print(rescaledX[0:5,:])
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=.3,
                                                    shuffle=True,
                                                    random_state=seed)
train2_x, test2_x, train2_y, test2_y = train_test_split(rescaledX, Y,
                                                        test_size=.3,
                                                        shuffle=True,
                                                        random_state=seed)
clf = MLPClassifier(max_iter=300, random_state=seed)
clf = SVC(random_state=seed)
clf.fit(train_x, train_y)
score = clf.score(test_x, test_y)
print(score)
clf.fit(train2_x, train2_y)
score = clf.score(test2_x, test2_y)
print(score)
