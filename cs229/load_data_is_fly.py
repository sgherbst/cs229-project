import os
import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt

from itertools import chain

from glob import glob

from cs229.files import top_dir
from cs229.annotation import Annotation
from cs229.image import img_to_mask
from cs229.contour import contour_label, find_contours

import joblib

CATEGORIES = ['neither', 'one', 'both']
FEATURES = ['area']
X_JOBLIB_NAME = 'X_is_fly.joblib'
Y_JOBLIB_NAME = 'y_is_fly.joblib'

def make_features(contour):
    # generate features
    features = []
    features.append(cv2.contourArea(contour))

    features = np.array(features, dtype=float)

    return features

def load_data():
    folders = ['12-04_17-54-43', '12-05-12-43-00', '12-07_16_45_00']
    folders = [os.path.join(top_dir(), 'images', folder) for folder in folders]
    folders = [glob(os.path.join(folder, '*.json')) for folder in folders]

    files = chain(*folders)

    X = []
    y = []

    for f in files:
        anno = Annotation(f)
        img = cv2.imread(anno.image_path, 0)

        mask = img_to_mask(img)
        contours = find_contours(img, mask=mask)

        for contour in contours:
            label = contour_label(anno, contour)
            if label in ['male', 'female']:
                label = 'one'
            if label is not None:
                X.append(make_features(contour))
                y.append(CATEGORIES.index(label))

    # assemble features
    X = np.array(X, dtype=float)

    # assemble labels
    y = np.array(y, dtype=int)

    return X, y

def main():
    X, y = load_data()

    joblib.dump(X, X_JOBLIB_NAME)
    joblib.dump(y, Y_JOBLIB_NAME)

if __name__ == '__main__':
    main()