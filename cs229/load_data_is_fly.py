import cv2
import numpy as np
import matplotlib.pyplot as plt

from cs229.files import get_annotation_files
from cs229.annotation import Annotation
from cs229.image import img_to_mask
from cs229.contour import contour_label, find_core_contours
from cs229.util import report_labels

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
    X = []
    y = []

    img_count = 0

    for f in get_annotation_files():
        img_count += 1

        anno = Annotation(f)
        img = cv2.imread(anno.image_path, 0)

        mask = img_to_mask(img)
        contours = find_core_contours(img, mask=mask)

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

    print('Used {} annotated images.'.format(img_count))
    report_labels(CATEGORIES, y)

    return X, y

def main():
    X, y = load_data()

    joblib.dump(X, X_JOBLIB_NAME)
    joblib.dump(y, Y_JOBLIB_NAME)

if __name__ == '__main__':
    main()