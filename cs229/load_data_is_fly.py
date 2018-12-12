import cv2
import numpy as np
from tqdm import tqdm

from cs229.files import get_file
from cs229.annotation import get_annotations
from cs229.image import img_to_mask
from cs229.contour import contour_label, find_contours
from cs229.util import report_labels_classification

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

    for anno in tqdm(get_annotations()):
        img = cv2.imread(anno.image_path, 0)

        mask = img_to_mask(img)
        contours = find_contours(img, mask=mask, type='core')

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

    report_labels_classification(y, CATEGORIES)

    return X, y

def main():
    X, y = load_data()

    joblib.dump(X, get_file('output', 'data', X_JOBLIB_NAME))
    joblib.dump(y, get_file('output', 'data', Y_JOBLIB_NAME))

if __name__ == '__main__':
    main()