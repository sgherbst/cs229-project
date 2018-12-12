from sklearn.model_selection import train_test_split
from cs229.load_data_id import CATEGORIES, make_features, X_JOBLIB_NAME, Y_JOBLIB_NAME
from cs229.files import get_file
from cs229.util import report_model_classification

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import joblib

CLF_JOBLIB_NAME = 'clf_id.joblib'

class IdPredictor:
    def __init__(self):
        self.clf = joblib.load(get_file('output', 'models', CLF_JOBLIB_NAME))

    def predict(self, contour_1, contour_2, patch_1, patch_2):
        features = make_features(contour_1, contour_2, patch_1, patch_2).reshape(1, -1)
        label = self.clf.predict(features)[0]
        return CATEGORIES[label]

def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs'))
    clf = clf.fit(X_train, y_train)

    return clf, X_train, X_test, y_train, y_test

def train_once(X, y):
    clf, X_train, X_test, y_train, y_test = train(X, y)
    report_model_classification(clf, X_train, X_test, y_train, y_test)

    joblib.dump(clf, get_file('output', 'models', CLF_JOBLIB_NAME))

def main():
    X = joblib.load(get_file('output', 'data', X_JOBLIB_NAME))
    y = joblib.load(get_file('output', 'data', Y_JOBLIB_NAME))

    train_once(X, y)

if __name__ == '__main__':
    main()