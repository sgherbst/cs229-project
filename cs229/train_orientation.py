import os
import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from itertools import chain
from math import degrees

from glob import glob

from cs229.files import top_dir
from cs229.annotation import Annotation
from cs229.full_img import FullImage
from cs229.image import img_to_mask
from cs229.patch import ImagePatch
from cs229.train_contour_classifier import contour_label

CATEGORIES = ['normal', 'flipped']

def angle_diff(a, b):
    # https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
    return ((a-b) + np.pi) % (2*np.pi) - np.pi

def img_to_features(img):
    M = cv2.moments(img)

    names = []
    # names += ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03']
    #names += ['mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03']
    #names += ['nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03']

    names = ['nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03']

    features = [M[name] for name in names]

    # H = cv2.HuMoments(M).flatten()
    # features = H

    return features

def load_data(tol_radians=0.1):
    folders = ['12-04_17-54-43', '12-05-12-43-00']
    folders = [os.path.join(top_dir(), 'images', folder) for folder in folders]
    folders = [glob(os.path.join(folder, '*.json')) for folder in folders]

    files = chain(*folders)

    X = []
    y = []

    for f in files:
        anno = Annotation(f)
        img = cv2.imread(anno.image_path)

        mask = img_to_mask(img)
        full_image = FullImage(img, mask=mask)

        for contour in full_image.contours:
            if contour_label(anno, contour) == 'female' and anno.has('fa') and anno.has('fh'):
                # make patch, which will compute the angle from the image
                patch = ImagePatch(full_image.img, contour)

                # compute angle from labels
                fa = anno.get('fa')[0]
                fh = anno.get('fh')[0]
                label_angle = np.arctan2(fa[1] - fh[1], fh[0] - fa[0])

                # find out if the image is flipped or not
                diff = abs(angle_diff(patch.orig_angle, label_angle))
                if diff <= tol_radians:
                    label = 'normal'
                elif np.pi-tol_radians <= diff <= np.pi+tol_radians:
                    label = 'flipped'
                else:
                    anno.warn('Could not properly determine whether image is flipped (diff={:0.1f} degrees)'.format(degrees(diff)))
                    plt.imshow(patch.img)
                    plt.show()
                    continue

                X.append(img_to_features(patch.img))
                y.append(CATEGORIES.index(label))

                print(CATEGORIES.index(label))
                plt.imshow(patch.img)
                plt.show()

                X.append(img_to_features(patch.flipped()))
                y.append(1-CATEGORIES.index(label))

                print(1-CATEGORIES.index(label))
                plt.imshow(patch.flipped())
                plt.show()

    # assemble features
    X = np.array(X).astype(float)

    # assemble labels
    y = np.array(y).astype(int)

    return X, y

def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scaler = StandardScaler()
    scaler.fit(X_train)

    print(scaler.transform(X_train)[:10,:])
    print(y_train)

    clf = LogisticRegression()
    clf = clf.fit(scaler.transform(X_train), y_train)

    y_pred = clf.predict(scaler.transform(X_test))

    print(classification_report(y_test, y_pred, target_names=CATEGORIES))

if __name__ == '__main__':
    main()