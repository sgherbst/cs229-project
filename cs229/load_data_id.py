import cv2
import numpy as np
import matplotlib.pyplot as plt

from cs229.files import get_annotation_files
from cs229.annotation import Annotation
from cs229.image import img_to_mask
from cs229.contour import contour_label, find_core_contours
from cs229.patch import crop_to_contour
from cs229.util import report_labels

import joblib

CATEGORIES = ['mf', 'fm']
X_JOBLIB_NAME = 'X_id.joblib'
Y_JOBLIB_NAME = 'y_id.joblib'

def make_features(contour_1, contour_2, patch_1, patch_2):
    # create area features
    area_1 = cv2.contourArea(contour_1)
    area_2 = cv2.contourArea(contour_2)

    # create eccentricity features
    aspect_1 = patch_1.estimate_aspect_ratio()
    aspect_2 = patch_2.estimate_aspect_ratio()

    # build feature vector
    features = []
    features.append(area_1)
    features.append(area_2)
    features.append(aspect_1)
    features.append(aspect_2)

    features = np.array(features, dtype=float)

    return features

def load_data():
    X = []
    y = []

    img_count = 0

    for f in get_annotation_files():
        anno = Annotation(f)
        img = cv2.imread(anno.image_path, 0)

        mask = img_to_mask(img)
        contours = find_core_contours(img, mask=mask)

        male = None
        female = None
        ok = True

        # sift through contours to find one corresponding to male and the other to female
        for contour in contours:
            label = contour_label(anno, contour)
            if label == 'male':
                if male is None:
                    male = contour
                else:
                    ok = False
                    anno.warn('Found two males, skipping...')
                    break
            elif label == 'female':
                if female is None:
                    female = contour
                else:
                    ok = False
                    anno.warn('Found two females, skipping...')
                    break

        if (male is not None) and (female is not None) and ok:
            male_patch = crop_to_contour(img, male)
            female_patch = crop_to_contour(img, female)

            X.append(make_features(male, female, male_patch, female_patch))
            y.append(CATEGORIES.index('mf'))

            X.append(make_features(female, male, female_patch, male_patch))
            y.append(CATEGORIES.index('fm'))

            img_count += 1

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