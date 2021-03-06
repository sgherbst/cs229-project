from sklearn import tree
from sklearn.model_selection import train_test_split
from cs229.load_data_is_fly import CATEGORIES, FEATURES, make_features, X_JOBLIB_NAME, Y_JOBLIB_NAME
from cs229.files import get_file
from cs229.util import report_model_classification

import joblib

CLF_JOBLIB_NAME = 'clf_is_fly.joblib'

class IsFlyPredictor:
    def __init__(self):
        self.clf = joblib.load(get_file('output', 'models', CLF_JOBLIB_NAME))

    def predict(self, contour):
        features = make_features(contour).reshape(1, -1)
        label = self.clf.predict(features)[0]
        return CATEGORIES[label]

def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    return clf, X_train, X_test, y_train, y_test

def train_once(X, y):
    clf, X_train, X_test, y_train, y_test = train(X, y)
    report_model_classification(clf, X_train, X_test, y_train, y_test)

    tree.export_graphviz(clf, out_file=get_file('output', 'diagrams', 'tree_is_fly.dot'), feature_names=FEATURES,
                         class_names=CATEGORIES, filled=True, rounded=True, special_characters=True)

    joblib.dump(clf, get_file('output', 'models', CLF_JOBLIB_NAME))

def main():
    X = joblib.load(get_file('output', 'data', X_JOBLIB_NAME))
    y = joblib.load(get_file('output', 'data', Y_JOBLIB_NAME))

    train_once(X, y)

if __name__ == '__main__':
    main()