import os.path
import numpy as np

from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from cs229.load_data_contour import CATEGORIES, contour_to_features
from cs229.files import top_dir

import joblib

CLF_JOBLIB_NAME = 'clf_contour.joblib'

class ContourPredictor:
    def __init__(self):
        self.clf = joblib.load(os.path.join(top_dir(), 'cs229', CLF_JOBLIB_NAME))

    def predict(self, contour):
        features = contour_to_features(contour).reshape(1, -1)
        label = self.clf.predict(features)[0]
        return CATEGORIES[label]

def train(X, y, plot=True, dump=True, report=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    if plot:
        tree.export_graphviz(clf, out_file='tree_contour.dot', feature_names=['contourArea'],
                             class_names=CATEGORIES, filled=True, rounded=True, special_characters=True)

    if report:
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=CATEGORIES))

    if dump:
        joblib.dump(clf, CLF_JOBLIB_NAME)

def main():
    X = joblib.load('X_contour.joblib')
    y = joblib.load('y_contour.joblib')

    train(X, y)

if __name__ == '__main__':
    main()