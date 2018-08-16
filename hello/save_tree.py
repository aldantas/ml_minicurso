from sklearn import tree

def save_tree(classifier, dataset):
    tree.export_graphviz(classifier, out_file='tree.dot',
                         feature_names=dataset.feature_names,
                         class_names=dataset.target_names, filled=True,
                         rounded=True, special_characters=True)
