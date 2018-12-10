import os.path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from cs229.files import top_dir
from cs229.load_data_wing import make_hog_patch, make_hog, patch_to_features
from cs229.util import train_experiment_regression, report_model_regression

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

def plot_pca(clf, X, y):
    X_t = clf.named_steps['pca'].transform(X)

    plt.scatter(X_t[:, 0].flatten(), y.flatten())

    plt.title('PCA for Wing Features')
    plt.xlabel('PCA 1')
    plt.ylabel('Wing angle (deg)')
    plt.grid()

    plt.savefig('pca_wing.eps'.format(type), bbox_inches='tight')

def train(X, y, plot=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = make_pipeline(PCA(n_components=6), PolynomialFeatures(degree=3), LinearRegression())
    clf.fit(X_train, y_train)

    if plot:
        plot_pca(clf, X_train, np.degrees(y_train))

    y_pred = clf.predict(X_test)

    return clf, np.degrees(y_test), np.degrees(y_pred)

def train_once(X, y):
    clf, y_test, y_pred = train(X, y, plot=True)

    report_model_regression(y_test, y_pred, 'degrees')

    joblib.dump(clf, CLF_JOBLIB_NAME)

def main():
    X = joblib.load('X_wing.joblib')
    y = joblib.load('y_wing.joblib')

    #train_once(X, y)
    train_experiment_regression(lambda: train(X, y), 'degrees')

if __name__ == '__main__':
    main()