import os.path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from cs229.files import top_dir
from cs229.load_data_orientation import CATEGORIES, make_hog_patch, make_hog, patch_to_features
from cs229.patch import crop_to_contour

import joblib

def clf_joblib_name(type):
    return 'clf_orient_{}.joblib'.format(type)

class PosePredictor:
    def __init__(self, type):
        self.clf = joblib.load(os.path.join(top_dir(), 'cs229', clf_joblib_name(type)))
        self.hog = make_hog()

    def predict(self, img, contour):
        patch = crop_to_contour(img, contour)

        # find center
        center = patch.estimate_center()
        center = (center[0]+patch.ulx, center[1]+patch.uly)

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

def plot_pca(clf, X, y):
    scaler = clf.named_steps['standardscaler']
    pca_std = clf.named_steps['pca']

    X_t = pca_std.transform(scaler.transform(X))

    for l, c, m in zip(range(2), ('blue', 'red'), ('o', 'x')):
        plt.scatter(X_t[y == l, 0], X_t[y == l, 1], color=c, label='class %s' % l, alpha=0.5, marker=m)

    plt.legend(loc='upper right')

    plt.grid()
    plt.show()

def train(X, y, type, plot=True, dump=True, report=True):
    X_train, X_test, y_train, y_test = train_test_split(X[type], y[type])

    clf = make_pipeline(StandardScaler(), PCA(n_components=3), LogisticRegression())

    clf.fit(X_train, y_train)

    if plot:
        plot_pca(clf, X_train, y_train)

    if report:
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=CATEGORIES))

    if dump:
        joblib.dump(clf, clf_joblib_name(type))

def main():
    X = joblib.load('X_orient.joblib')
    y = joblib.load('y_orient.joblib')

    print('Training orientation detector for male fly...')
    train(X, y, 'male')

    print('Training orientation detector for female fly...')
    train(X, y, 'female')

if __name__ == '__main__':
    main()