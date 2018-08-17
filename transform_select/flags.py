# Ler o dataset
import pandas as pd
from utils import *
filename = 'dataset/flag.csv'
data = pd.read_csv(filename, names=column_names)
pd.set_option('display.max_columns', 30)
print(data[0:5])

labels = data['religion'].values
del data['religion']
del data['name']

# Converter valores nominais para inteiros
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
encoded_data = data.copy()
for column in nominal_columns:
    encoded_data[column] = le.fit_transform(data[column])
    print(data[0:5])
run_clf(encoded_data, labels, 'LabelEncoder:')

# Converter valores categoricos para sequências de bits
from sklearn.preprocessing import OneHotEncoder
idxs = [data.columns.get_loc(col) for col in categorical_columns]
oneh = OneHotEncoder(categorical_features=idxs, sparse=False)
onehot_data = oneh.fit_transform(encoded_data)
print(data.shape)
print(onehot_data.shape)
print(onehot_data[0])
run_clf(onehot_data, labels, 'OneHotEncoder:')

# Reduzir dimensionalidade
from sklearn.feature_selection import SelectKBest
sel = SelectKBest(k=10)
reduced_data = sel.fit_transform(onehot_data, labels)
run_clf(reduced_data, labels, 'Feature Selection:')
