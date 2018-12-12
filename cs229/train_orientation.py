import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from cs229.files import get_file
from cs229.load_data_orientation import (CATEGORIES, make_hog_patch, make_hog, patch_to_features,
                                         X_JOBLIB_NAME, Y_JOBLIB_NAME)
from cs229.util import report_model_classification

import joblib

def clf_joblib_name(type):
    return 'clf_orient_{}.joblib'.format(type)

class PosePredictor:
    def __init__(self, type):
        self.clf = joblib.load(get_file('output', 'models', clf_joblib_name(type)))
        self.hog = make_hog()

    def predict(self, patch):
        # find center
        center = patch.estimate_center(absolute=True)

        # find angle
        angle = patch.estimate_angle()

        # figure out if the angle needs to be flipped 180
        hog_patch = make_hog_patch(patch)
        features = patch_to_features(hog_patch, self.hog).reshape(1, -1)
        label = self.clf.predict(features)[0]
        category = CATEGORIES[label]

        # apply 180 degree correction
        if category == 'normal':
            pass
        elif category == 'flipped':
            angle += np.pi
        else:
            raise Exception('Invalid category.')

        # return result
        return center, angle

def plot_pca(X, y, type):
    # apply PCA to data
    pca = PCA(n_components=200)
    pca.fit(X, y)

    # plot variance explained by each component
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of PCA components')
    plt.ylabel('Total explained variance ratio')
    plt.grid()

    plt.savefig(get_file('output', 'graphs', 'pca_orient_{}_explained.eps'.format(type)), bbox_inches='tight')
    plt.clf()

    # show how data are separated by first two principal components
    X_t = pca.transform(X)

    for l, c, m in zip(range(2), ('blue', 'red'), ('o', 'x')):
        plt.scatter(X_t[y == l, 0], X_t[y == l, 1], color=c, label=CATEGORIES[l], alpha=0.5, marker=m)

    plt.legend(loc='upper right')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.grid()

    plt.savefig(get_file('output', 'graphs', 'pca_orient_{}.eps'.format(type)), bbox_inches='tight')
    plt.clf()

def train(X, y, type, plot=False):
    X_train, X_test, y_train, y_test = train_test_split(X[type], y[type])

    clf = make_pipeline(PCA(n_components=15), LogisticRegression(solver='lbfgs'))

    clf.fit(X_train, y_train)

    if plot:
        plot_pca(X_train, y_train, type)

    return clf, X_train, X_test, y_train, y_test

def train_once(X, y, type):
    clf, X_train, X_test, y_train, y_test = train(X, y, type, plot=True)
    report_model_classification(clf, X_train, X_test, y_train, y_test)

    joblib.dump(clf, get_file('output', 'models', clf_joblib_name(type)))

def main():
    X = joblib.load(get_file('output', 'data', X_JOBLIB_NAME))
    y = joblib.load(get_file('output', 'data', Y_JOBLIB_NAME))

    for type in ['male', 'female']:
        print('{} fly orientation detector...'.format(type.capitalize()))
        train_once(X, y, type)

if __name__ == '__main__':
    main()