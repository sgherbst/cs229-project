import os.path
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from cs229.load_data_id import CATEGORIES, FEATURES, make_features, X_JOBLIB_NAME, Y_JOBLIB_NAME
from cs229.files import top_dir

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import joblib

CLF_JOBLIB_NAME = 'clf_id.joblib'

class IdPredictor:
    def __init__(self):
        self.clf = joblib.load(os.path.join(top_dir(), 'cs229', CLF_JOBLIB_NAME))

    def predict(self, contour_1, contour_2, patch_1, patch_2, img):
        features = make_features(contour_1, contour_2, patch_1, patch_2, img).reshape(1, -1)
        label = self.clf.predict(features)[0]
        return CATEGORIES[label]

def plot_pca(clf, X, y):
    scaler = clf.named_steps['standardscaler']
    pca_std = clf.named_steps['pca']

    X_t = pca_std.transform(scaler.transform(X))

    for l, c, m in zip(range(2), ('blue', 'red'), ('o', 'x')):
        plt.scatter(X_t[y == l, 0], X_t[y == l, 1], color=c, label='class %s' % l, alpha=0.5, marker=m)

    plt.legend(loc='upper right')

    plt.grid()
    plt.show()

def train(X, y, plot=True, dump=True, report=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    #clf = tree.DecisionTreeClassifier()
    #clf = make_pipeline(StandardScaler(), PCA(n_components=4), LogisticRegression())
    clf = make_pipeline(StandardScaler(), LogisticRegression())
    clf = clf.fit(X_train, y_train)

    if plot:
        # tree.export_graphviz(clf, out_file='tree_id.dot', feature_names=FEATURES,
        #                      class_names=CATEGORIES, filled=True, rounded=True, special_characters=True)
        #plot_pca(clf, X_train, y_train)
        pass

    if report:
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=CATEGORIES))

    if dump:
        joblib.dump(clf, CLF_JOBLIB_NAME)

def main():
    X = joblib.load(X_JOBLIB_NAME)
    y = joblib.load(Y_JOBLIB_NAME)

    train(X, y)

if __name__ == '__main__':
    main()