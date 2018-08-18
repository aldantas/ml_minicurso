import pandas as pd

def load_dataset(filename):
    df = pd.read_csv(filename, header=None)
    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1]
    return X, Y

# filename = '../transform_select/dataset/flag.csv'
# print(load_dataset(filename))
