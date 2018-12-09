import os.path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from cs229.files import top_dir
from cs229.load_data_wing import CATEGORIES, make_hog_patch, make_hog, patch_to_features
from cs229.patch import crop_to_contour
from cs229.util import train_experiment_regression, report_model_regression

import joblib

CLF_JOBLIB_NAME = 'clf_wing.joblib'

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

    clf = make_pipeline(PCA(n_components=3), PolynomialFeatures(degree=3), LinearRegression())
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