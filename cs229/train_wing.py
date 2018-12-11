import os.path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from cs229.files import top_dir
from cs229.load_data_wing import make_hog_patch, make_hog, patch_to_features
from cs229.util import report_model_regression

import joblib

CLF_JOBLIB_NAME = 'clf_wing.joblib'

class WingPredictor:
    def __init__(self):
        self.clf = joblib.load(os.path.join(top_dir(), 'cs229', CLF_JOBLIB_NAME))
        self.hog = make_hog()

    def predict(self, patch):
        # create a hog patch for the right wing
        hog_patch_right = make_hog_patch(patch)
        features_right = patch_to_features(hog_patch_right, self.hog).reshape(1, -1)
        wing_angle_right = self.clf.predict(features_right)[0]

        # create a hog patch for the left wing
        hog_patch_left = make_hog_patch(patch.flip('horizontal'))
        features_left = patch_to_features(hog_patch_left, self.hog).reshape(1, -1)
        wing_angle_left = self.clf.predict(features_left)[0]

        return wing_angle_right, wing_angle_left

def plot_pca(X, y):
    # apply PCA to data
    pca = PCA(n_components=200)
    pca.fit(X, y)

    # plot variance explained by each component
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of PCA components')
    plt.ylabel('Total explained variance ratio')
    plt.grid()

    plt.savefig('pca_wing_explained.eps'.format(type), bbox_inches='tight')
    plt.clf()

    # plot principal component vs angle
    X_t = pca.transform(X)

    plt.scatter(X_t[:, 0].flatten(), y.flatten())
    plt.xlabel('PCA Component 1')
    plt.ylabel('Wing angle (deg)')
    plt.grid()

    plt.savefig('pca_wing.eps'.format(type), bbox_inches='tight')
    plt.clf()

def train(X, y, plot=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = make_pipeline(PCA(n_components=40), LinearRegression())
    clf.fit(X_train, y_train)

    if plot:
        plot_pca(X_train, np.degrees(y_train))

    return clf, X_train, X_test, y_train, y_test

def train_once(X, y):
    clf, X_train, X_test, y_train, y_test = train(X, y, plot=True)

    report_model_regression('radians', clf, X_train, X_test, y_train, y_test)

    joblib.dump(clf, CLF_JOBLIB_NAME)

def main():
    X = joblib.load('X_wing.joblib')
    y = joblib.load('y_wing.joblib')

    train_once(X, y)
    #train_experiment_regression(lambda: train(X, y), 'degrees')

if __name__ == '__main__':
    main()