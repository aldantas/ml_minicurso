from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

column_names = ['name','landmass','zone','area','population','language','religion','bars','stripes','colours','red','green','blue','gold','white','black','orange','mainhue','circles','crosses','saltires','quarters','sunstars','crescent','triangle','icon','animate','text','topleft','botright']
numeric_columns = ['area', 'population', 'bars', 'stripes', 'colours',
                   'circles', 'crosses', 'saltires', 'quarters', 'sunstars']
categorical_columns = ['zone', 'white', 'landmass', 'icon', 'orange', 'red',
                       'language', 'crescent', 'animate', 'botright', 'gold',
                       'topleft', 'text', 'triangle', 'blue', 'black', 'green',
                       'mainhue']
nominal_columns = ['mainhue','topleft','botright']

def run_clf(data, labels, text=''):
    clf = Perceptron(max_iter=1000, random_state=42)
    # clf = RandomForestClassifier(random_state=42)
    train_data,test_data,train_labels,test_labels = train_test_split(
        data,labels,test_size=0.3, shuffle=True, random_state=42)
    clf.fit(train_data,train_labels)
    predicted_labels = clf.predict(test_data)
    score = clf.score(test_data, test_labels)
    print(text, score)
