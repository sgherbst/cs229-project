import os.path
import numpy as np

from sklearn import tree
from sklearn.model_selection import train_test_split
from cs229.load_data_is_fly import CATEGORIES, FEATURES, make_features, X_JOBLIB_NAME, Y_JOBLIB_NAME
from cs229.files import top_dir
from cs229.util import train_experiment, report_model

import joblib

CLF_JOBLIB_NAME = 'clf_is_fly.joblib'

class IsFlyPredictor:
    def __init__(self):
        self.clf = joblib.load(os.path.join(top_dir(), 'cs229', CLF_JOBLIB_NAME))

    def predict(self, contour):
        features = make_features(contour).reshape(1, -1)
        label = self.clf.predict(features)[0]
        return CATEGORIES[label]

def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    return clf, y_test, y_pred

def train_once(X, y):
    clf, y_test, y_pred = train(X, y)
    report_model(y_test, y_pred, CATEGORIES)

    tree.export_graphviz(clf, out_file='tree_is_fly.dot', feature_names=FEATURES,
                         class_names=CATEGORIES, filled=True, rounded=True, special_characters=True)

    joblib.dump(clf, CLF_JOBLIB_NAME)

def main():
    X = joblib.load(X_JOBLIB_NAME)
    y = joblib.load(Y_JOBLIB_NAME)

    train_once(X, y)
    #train_experiment(lambda: train(X, y), CATEGORIES)

if __name__ == '__main__':
    main()